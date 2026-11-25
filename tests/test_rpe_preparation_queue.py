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

    def test_resume_retires_only_unclaimed_incomplete_max_width_delivery(self):
        from copy import deepcopy
        from trottertracks.rpe_optimizer_concurrent import stale_max_width_readiness, max_width_signal_missing_frozen_input
        root=str(ROOT);request={'preparation':dict(provider='DF_SIGNALS',controller_root=root,
            command=['python','-B','runner','--input',str(ROOT/'fixture.json'),'--inputs-commit','C','--transport-source','M'],
            inputs={'private/df-terminal-transport/src/trottertracks/pr_df_max_width.py':'fixture'},input_hashes=[])}
        good=deepcopy(request);good['preparation']['input_hashes']=[dict(commit='C',path='fixture.json')]
        self.assertTrue(max_width_signal_missing_frozen_input(request));self.assertFalse(max_width_signal_missing_frozen_input(good))
        pending={k:dict(request=good if k=='fixed' else request) for k in ('unreserved','cancelled','queued','running','assigned','attempted','fixed')}
        kinds=dict(cancelled='CANCELLED',queued='QUEUED',running='RUNNING',assigned='ASSIGNED',attempted='QUEUED',fixed='QUEUED')
        states={k:('PREPARATION:signal',v)for k,v in kinds.items()}
        rows={k:dict(method='PREPARATION:signal',attempt_id='OWNER' if k in ('running','assigned','attempted') else None)for k in kinds}
        calls=[];store=SimpleNamespace(row=rows.get,cancel=lambda m,j:calls.append(j)or True)
        self.assertEqual(stale_max_width_readiness(store,pending,states),['unreserved','cancelled','queued'])
        self.assertEqual(calls,['queued'])

    def test_max_width_bundle_alias_reuses_one_frozen_request_across_resume(self):
        from copy import deepcopy
        from trottertracks.rpe_optimizer_concurrent import max_width_signal_key, max_width_signal_owner
        from trottertracks.rpe_preparation_queue import normalize_preparation_request
        request = self.request('signal', 'same-science')
        request['original_contract'] = {'rounds': [1, 2]}
        blobs = {hashlib.sha256(body).hexdigest(): base64.b64encode(body).decode()
                 for body in (b'mapper', b'input', b'bundle-one', b'bundle-two')}
        mapper_sha, input_sha, bundle_one, bundle_two = blobs
        prep = request['preparation']
        prep.update(provider='DF_SIGNALS', controller_root=str(ROOT),
            command=['ARTIFICIAL_ONLY','--input',str(ROOT/'fixture.json'),'--inputs-commit','c'*40,
                     '--transport-source','m'*40],
            inputs={'private/df-terminal-transport/src/trottertracks/pr_df_max_width.py':mapper_sha,
                    'workspace/fixture.json':input_sha},
            input_hashes=[dict(commit='c'*40,path='fixture.json',sha256=input_sha)],
            input_git_bundles=[{'sha256':bundle_one}])
        alias = deepcopy(request)
        alias['preparation']['input_git_bundles'] = [{'sha256':bundle_two}]
        # The same bytes may also be referenced by a later Git commit.
        alias['preparation']['command'][alias['preparation']['command'].index('--inputs-commit')+1] = 'f'*40
        alias['preparation']['input_hashes'][0]['commit'] = 'f'*40
        first = self.executor()
        first.artifact_cache.put(blobs)
        one = first.submit_preparation(request, approved=True, budget=1, contract=request['original_contract'])
        normalized = normalize_preparation_request(alias, source_manifest=first.source_manifest)
        self.assertNotEqual(one, normalized['request_id'])
        self.assertEqual(max_width_signal_key(first.store.row(one)['request']), max_width_signal_key(normalized))
        self.assertEqual(first.submit_preparation(alias, approved=True, budget=1, contract=request['original_contract']), one)
        self.assertEqual(max_width_signal_owner(first.store, normalized)['job_id'], one)
        first.state.retain_queued_preparation = True
        self.close_one(first)
        second = self.executor()
        self.assertEqual(second.submit_preparation(alias, approved=True, budget=1, contract=request['original_contract']), one)
        self.assertEqual(second.store.preparation_reservations(), {'signal': 1})
        assignment, = self.poll_host(second, 'server1')['assignments']
        self.assertEqual(assignment['job_id'], one)
        second.rpc('/v1/complete', self.prep_complete(second, assignment))
        second.adopt_preparation(one)
        self.assertEqual(second.submit_preparation(alias, approved=True, budget=1, contract=request['original_contract']), one)
        second.adopt_preparation(one)
        events = second.store.events()
        self.assertEqual(sum(row['event'] == 'ASSIGN' for row in events), 1)
        self.assertEqual(sum(row['event'] == 'ADOPT' for row in events), 1)

    def test_planning_boundary_retains_preparation_and_reuses_reservation(self):
        first = self.executor()
        job = self.submit(first, 'signal', 'independent', budget=1)
        first.state.retain_queued_preparation = True
        self.close_one(first)
        self.assertEqual(first.queued_preparation_count(), 1)
        second = self.executor()
        self.assertEqual(self.submit(second, 'signal', 'independent', budget=1), job)
        assignment, = self.poll_host(second, 'server1')['assignments']
        self.assertEqual(assignment['job_id'], job)
        second.rpc('/v1/complete', self.prep_complete(second, assignment))
        self.assertEqual(second.queued_preparation_count(), 0)
        self.assertEqual(second.store.preparation_reservations(), {'signal': 1})
        self.assertEqual(sum(row['event'] == 'ASSIGN' for row in second.store.events()), 1)

    def test_horizon_round_shares_existing_queue_and_stage_budgets(self):
        from trottertracks.rpe_preparation_queue import submit_horizon_preparation, submit_registered_horizons
        from trottertracks.rpe_scaled_shots import projection_id

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
        requests = [dict(source_configuration_id=cid, configuration_id=projection_id(cid, 12, 13), M=13,
                         source_M=12, extension_rounds=[13]) for cid in bindings]
        candidates = [dict(configuration_id=cid) for cid in bindings]
        reports = submit_registered_horizons(runtime, candidates, {'M': [11, 12, 13]}, 12)
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

    def test_model_priced_pf_horizon_queues_signal_only(self):
        from trottertracks.rpe_preparation_queue import submit_horizon_preparation

        executor = self.executor(capacity=2)
        tasks = {}
        for name, stage, axes in (('signal', 'signal', []), ('pair', 'compile', ['X', 'Y'])):
            task = self.request(stage, name)
            task['original_contract'] = dict(round=13, physical_id='terminal')
            task['preparation'].update(horizon_round={'m': 13, 'physical_id': 'terminal'}, price_axes=axes)
            tasks[name] = {'request': task}
        binding = {'source_M': 12, 'rounds': {'13': {'physical_id': 'terminal',
                   'signal': 'signal', 'X': 'pair', 'Y': 'pair'}}}
        spec = dict(horizon_extensions=dict(bindings={'parent': binding}, tasks=tasks),
                    pf_terminal_price_leaves={'parent': {'physical_id': 'terminal'}},
                    budgets={'signal': 1, 'compile': 1})
        runtime = SimpleNamespace(settings=spec, executor=executor, approved=True)
        extension = dict(source_configuration_id='parent', configuration_id='extended',
                         M=13, source_M=12, extension_rounds=[13])
        report, = submit_horizon_preparation(runtime, [extension])
        self.assertEqual(set(report['jobs']['13']), {'signal'})
        self.assertEqual(report['incomplete']['13'], dict.fromkeys(('X', 'Y'), 'HORIZON_PRICE_FROM_ADOPTED_PARENT_MODEL'))
        self.assertEqual(executor.store.preparation_reservations(), {'signal': 1})

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

    def test_horizon_ledger_preserves_rows_and_reports_actual_queue_progress(self):
        from trottertracks.rpe_preparation_queue import save_horizon_preparation

        executor = self.executor(capacity=1)
        job_id = self.submit(executor, 'signal', 'terminal')
        output = self.run / 'HORIZON_PREPARATION.json'
        first = dict(source_configuration_id='first', M=13, jobs={'13': {'signal': job_id}})
        second = dict(source_configuration_id='other', M=13, jobs={})
        save_horizon_preparation(output, [first], executor.store)
        assignment, = self.poll_host(executor,'server1')['assignments']
        executor.rpc('/v1/complete', self.prep_complete(executor, assignment))
        save_horizon_preparation(output, [second], executor.store)
        records = {row['source_configuration_id']: row for row in json.loads(output.read_bytes())}
        self.assertEqual(set(records), {'first','other'})
        progress = records['first']['job_progress']['13']['signal']
        self.assertEqual(progress['state'], 'ACCEPTED')
        self.assertEqual(progress['outcome_status'], 'PREPARATION_COMPLETE')
        self.assertFalse(progress['adopted'])

    def test_horizon_price_status_uses_the_matching_audited_terminal_axes(self):
        from trottertracks.rpe_preparation_queue import save_horizon_preparation

        executor = self.executor(capacity=1)
        output = self.run/'HORIZON_PREPARATION.json'
        terminal = self.run/'pf_auto'/'TERMINAL_PRICES.json'
        terminal.parent.mkdir()
        row = dict(source_configuration_id='parent', M=13, jobs={}, terminal_physical_id='round13',
                   status='HORIZON_PROVIDER_CONTRACT_HOLD', incomplete={'13': {
                       'signal': 'WAITING', 'X': 'PROVIDER', 'Y': 'PROVIDER'}})
        ready = dict(status='READY', entry={'prices': {'commit': 'fixed', 'path': 'prices.json'}},
                     verified_terminal_axes=dict(m=13, physical_id='wrong', axes=['X', 'Y']))
        terminal.write_text(json.dumps({'parent': ready}))
        save_horizon_preparation(output, [row], executor.store)
        self.assertIn('X', json.loads(output.read_text())[0]['incomplete']['13'])
        ready['verified_terminal_axes']['physical_id'] = 'round13'
        terminal.write_text(json.dumps({'parent': ready}))
        save_horizon_preparation(output, [], executor.store)
        result = json.loads(output.read_text())[0]
        self.assertEqual(result['terminal_prices']['status'], 'READY')
        self.assertEqual(result['incomplete'], {'13': {'signal': 'WAITING'}})

    def test_horizon_ledger_distinguishes_missing_parent_and_collected_signal(self):
        from trottertracks.rpe_preparation_queue import save_horizon_preparation
        from trottertracks.rpe_release_registry import write

        output = self.run/'HORIZON_PREPARATION.json'
        write(self.run/'PF_PREPARATION_STATUS.json', {'records': {'parent': {
            'status': 'PREPARATION_WAITING', 'reason': 'Original stage still owned'}}})
        row = dict(source_configuration_id='parent', M=13, jobs={},
                   status='HORIZON_PROVIDER_NOT_CONFIGURED')
        save_horizon_preparation(output, [row], None)
        result, = json.loads(output.read_bytes())
        self.assertEqual(result['status'], 'WAITING_FOR_PARENT_SIGNAL')
        self.assertEqual(result['preparation_detail']['reason'], 'Original stage still owned')

        write(self.run/'pf_auto/horizon/parent/PREPARATION.json', dict(status='WAITING_FOR_PARENT_PRICES'))
        save_horizon_preparation(output, [row], None)
        result, = json.loads(output.read_bytes())
        self.assertEqual(result['status'], 'WAITING_FOR_PARENT_PRICES')

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
        from concurrent.futures import ThreadPoolExecutor
        from copy import deepcopy
        from threading import Event
        from unittest.mock import patch
        from trottertracks import rpe_preparation_worker
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
        checking = Event()
        release = Event()
        validate = rpe_preparation_worker.validate_preparation_fingerprint

        def blocked_validator(*args):
            checking.set()
            if not release.wait(5):
                raise AssertionError('Another host could not poll during preparation verification')
            return validate(*args)

        with patch.object(rpe_preparation_worker, 'validate_preparation_fingerprint', blocked_validator), ThreadPoolExecutor(3) as pool:
            pending = pool.submit(ex.rpc, '/v1/poll', report)
            try:
                self.assertTrue(checking.wait(2))
                self.assertEqual(pool.submit(self.poll_host, ex, 'local').result(2)['assignments'], [])
                # An overlapping poll may finish while the receipt is being
                # checked. Its newer offer must not be replaced by the old one.
                self.assertEqual(pool.submit(self.poll_host, ex, 'server1', 0).result(2)['assignments'], [])
            finally:
                release.set()
            self.assertEqual(pending.result(2)['assignments'], [])
        self.assertIsNone(ex.store.row(job))
        assigned = ex.rpc('/v1/poll', report)['assignments']
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
        def offer(host,policy,error=None,*,owned=None,current_poll=None):
            if threading.current_thread().name=='PREPARATION_METRICS_FIXTURE':
                inspecting.set()
                if not release.wait(3):raise AssertionError('Fixture did not release metrics')
            return original(host,policy,error,owned=owned,current_poll=current_poll)
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

    def test_scaled_selector_dispatches_independent_q3_while_first_candidate_is_waiting(self):
        from concurrent.futures import ThreadPoolExecutor
        from fractions import Fraction
        import time

        from trottertracks import rpe_method_optimizer as optimizer
        from trottertracks import rpe_scaled_shots as scaled

        ex = self.executor(capacity=2, controller_parallel=True)
        prep = self.submit(ex, 'signal', 'blocked-candidate')
        original, = self.poll_host(ex, 'server1')['assignments']
        self.assertEqual(original['job_id'], prep)
        ex.preparation_dependencies['blocked'] = [prep]
        view = ex.for_method('PR_PAULI')
        tau = Fraction(scaled.EPSILON)**2
        candidates = [dict(method='PR_PAULI', configuration_id=name, name=name,
                           M=0, key={}, ready=True, needs=[], prices=[price, price])
                      for name, price in [('blocked', 1), ('independent', 2)]]
        policy = scaled.settings(dict(M=[0], grid=dict(min_k=0, max_k=0, divisor=1), compare_paper=False))

        def claim(host):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                assignments = self.poll_host(ex, host)['assignments']
                if assignments:
                    task, = assignments
                    return task
                time.sleep(.01)
            self.fail('No artificial assignment for '+host)

        def complete(task, host):
            payload = self.completion(task)
            outcome = json.loads(base64.b64decode(payload['files']['OUTCOME.json']))
            outcome.update(MSE_lower=optimizer.enc(0), MSE_upper=optimizer.enc(tau/2))
            payload.update(host_id=host)
            payload['files']['OUTCOME.json'] = base64.b64encode(json.dumps(outcome).encode()).decode()
            self.assertFalse(ex.rpc('/v1/complete', payload)['duplicate'])
            self.assertTrue(ex.rpc('/v1/complete', payload)['duplicate'])

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(scaled.run, 'PR_PAULI', candidates, {}, view, dict(scaled_shots=policy), tau, 2,
                                 answer_now=view.answer_now, await_dispatch=view.await_dispatch,
                                 read_completed=view.read_scaled_completions)
            try:
                independent = claim('server2')
                self.assertEqual(independent['request']['configuration_id'], 'independent')
                complete(independent, 'server2')
                self.assertEqual(ex.store.counts('PR_PAULI'), 1)
                ex.rpc('/v1/complete', self.prep_complete(ex, original))
                blocked = claim('server1')
                self.assertEqual(blocked['request']['configuration_id'], 'blocked')
                complete(blocked, 'server1')
                result = future.result(timeout=5)
            finally:
                if not future.done():
                    (self.run/'STOP_REQUESTED.json').write_text('{}')
                    with ex.state.cv:
                        ex.state.cv.notify_all()
        self.assertEqual(result['stop_reason'], 'SCALED_SHOT_GRID_COMPLETE')
        q3_ids = {row['job_id'] for row in (independent, blocked)}
        for event in ('ASSIGN', 'ADOPT'):
            self.assertEqual(sum(r['event'] == event and r['job_id'] in q3_ids
                                 for r in ex.store.events()), 2)

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
        # Preparation is incomplete, not evidence that no feasible point exists.
        self.assertEqual(result['stop_reason'],'HOST_ELIGIBILITY_HOLD')
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
        for wait in (False, True):
            result=ex.preparation_result(two,wait=wait)
            self.assertEqual(result['status'],'PREPARATION_DEPENDENCY_HOLD')
        self.assertEqual(self.poll_host(ex,'server1')['assignments'],[])

    def test_follow_preparation_resumes_on_published_result_while_sibling_runs(self):
        from threading import Event, Thread
        from trottertracks.rpe_preparation_bridge import DistributedPreparationBridge

        executor = self.executor(capacity=2)
        bridge = DistributedPreparationBridge(executor, ROOT)
        jobs = {name: self.submit(executor, 'signal', name) for name in ('a', 'b')}
        assignments = {item['job_id']: item for item in self.poll_host(executor, 'server1', 2)['assignments']}
        observed = Event()
        watching = Event()
        stopped = Event()
        landed = []

        def advance(candidate):
            job = jobs[candidate]
            result = executor.preparation_result(job, adopt=False)
            if result is None:
                if candidate == 'b':
                    watching.set()
                return job
            self.assertEqual(result['status'], 'PREPARATION_COMPLETE')
            executor.adopt_preparation(job)
            landed.append(candidate)
            observed.set()

        waiter = Thread(target=bridge.follow_preparations,
                        args=(('a', 'b'), advance, stopped.is_set))
        waiter.start()
        try:
            self.assertTrue(watching.wait(3))
            receipt = self.prep_complete(executor, assignments[jobs['b']])
            executor.rpc('/v1/complete', receipt)
            self.assertTrue(observed.wait(3))
            self.assertEqual(landed, ['b'])
            self.assertIsNone(executor.store.row(jobs['a'])['completed_epoch'])
            self.assertTrue(executor.rpc('/v1/complete', receipt)['duplicate'])
        finally:
            stopped.set()
            executor.rpc('/v1/complete', self.prep_complete(executor, assignments[jobs['a']]))
            waiter.join(3)
        self.assertFalse(waiter.is_alive())
        bridge.follow_preparations(('a',), advance, lambda: False)
        self.assertEqual(landed, ['b', 'a'])
        self.assertEqual(sum(row['event'] == 'ASSIGN' for row in executor.store.events()), 2)

    def test_preparation_completion_watch_waits_for_publication_and_handles_lost_owner(self):
        from time import time
        from unittest.mock import patch

        executor = self.executor(capacity=2)
        completed = self.submit(executor, 'signal', 'accepted-not-yet-visible')
        lost = self.submit(executor, 'signal', 'contact-lost')
        assignments = {a['job_id']: a for a in self.poll_host(executor, 'server1', 2)['assignments']}
        payload = self.prep_complete(executor, assignments[completed])
        with patch.object(executor, '_publish_transfer', side_effect=OSError('fixture publish interruption')):
            with self.assertRaises(OSError):
                executor.rpc('/v1/complete', payload)
        watched = {completed: (), lost: ()}
        self.assertEqual(executor.ready_preparations(watched, wait=False, stop_requested=lambda: False), [])
        self.assertIsNone(executor.preparation_result(completed, adopt=False))

        executor.rpc('/v1/complete', payload)
        with patch('trottertracks.rpe_optimizer_concurrent.time.time', return_value=time()+600):
            executor._check_worker_contact()
        self.assertCountEqual(executor.ready_preparations(watched, wait=False, stop_requested=lambda: False), watched)
        self.assertEqual(executor.preparation_result(lost, adopt=False)['status'], 'UNKNOWN')
        self.assertEqual(executor.store.row(lost)['attempt_id'], assignments[lost]['attempt_id'])


# Base fixture's existing tests already have their own coverage; inherited
# methods here would rerun them needlessly.
for _name in tuple(vars(Base.ControllerFailures)):
    if _name.startswith('test_') and _name not in vars(PreparationQueue):
        setattr(PreparationQueue,_name,None)

if __name__=='__main__':unittest.main()
