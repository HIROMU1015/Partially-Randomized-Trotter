"""Preparation concurrency and exact reuse with artificial provider leaves."""
from concurrent.futures import ThreadPoolExecutor
import base64
from copy import deepcopy
from contextlib import nullcontext
import importlib.util
import json
import hashlib
import shutil
import subprocess
from pathlib import Path
from random import Random
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pr_candidate_completion as C
from trottertracks import rpe_preparation_bridge as B
from trottertracks.rpe_preparation_artifacts import ArtifactCache
from trottertracks.rpe_preparation_queue import normalize_preparation_request
from trottertracks import rpe_preparation_worker as W
from trottertracks.rpe_q3_store import JobStore


class CompletionBridgeTests(unittest.TestCase):
    def test_pf_redelivery_reuses_frozen_stage_under_changed_search_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            candidate, job, stage = 'fixed-candidate', 'original-job', 'PF_COMPILE'
            workspace = run/f'pf_auto/candidates/{candidate}/pf_auto/pf_completion/provider_workspace'
            directory = workspace/'docs/research/distributed_selected_pf'/candidate
            source = dict(commit='fixed', files={})
            req = normalize_preparation_request(dict(route=dict(kind='preparation'),
                original_contract={'calls': 2}, preparation=dict(source=source, provider=stage, stage='compile')))
            C.put(run/'pf_auto/WAVES.json', {candidate: dict(status='PREPARATION_INCOMPLETE',
                reason='PF_COMPLETION_REQUEST_MISMATCH', differing_fields=['exploratory_cost'])})
            C.put(workspace.parent/'PROVIDER_SOURCE.json', {'source': source['commit']})
            C.put(directory/(stage+'.DISTRIBUTED.json'), dict(job_id=job, request=req))
            stage_input = directory/(stage+'.INPUT.json')
            C.put(stage_input, dict(value=dict(configuration=dict(id=candidate))))
            proof = dict(kind='PF_PORTABLE_INPUT_DELIVERY', configuration_id=candidate,
                         audit_attempt='old', input_sha256=hashlib.sha256(stage_input.read_bytes()).hexdigest())
            executor = SimpleNamespace(run_dir=run, _user_stop_requested=lambda: False,
                                       store=SimpleNamespace(row=lambda _: dict(request=req)), log=lambda **_: None)
            bridge = B.DistributedPreparationBridge(executor, run,
                dict(pf_stage_audit_recoveries={job: proof}), approved=True)
            with patch.object(bridge, 'call') as call:
                bridge.reconcile_pf_input_redeliveries(dict(exploratory_cost=True), sys.executable)
                completion, cmd, target, provider, output = call.call_args.args
                self.assertEqual((completion.source, target, provider), ('fixed', directory, stage))
                self.assertEqual(cmd[cmd.index('--input')+1], str(stage_input))
                self.assertEqual(call.call_args.kwargs, {'wait': False})
                stage_input.write_text('{}')
                call.reset_mock()
                bridge.reconcile_pf_input_redeliveries(dict(exploratory_cost=True), sys.executable)
                call.assert_not_called()

    def test_pf_redelivery_needs_prelaunch_proof_and_a_changed_delivery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = 'PF_COMPILE'
            path = root/(stage+'.INPUT.json')
            C.put(path, dict(value=dict(configuration=dict(id='fixed-id'))))
            asset = 'workspace/case/'+path.name
            adapter = 'private/pf-transport/scripts/run_pf_preparation_stage.py'
            request = normalize_preparation_request(dict(route=dict(kind='preparation'),
                original_contract={'calls':2}, preparation=dict(stage='compile', provider=stage,
                    source=dict(commit='fixed',files={}), inputs={asset:hashlib.sha256(path.read_bytes()).hexdigest()})))
            job = request['request_id']
            row = dict(request=request, request_sha256=W.digest(request), state='ACCEPTED',
                       attempt_id='previous', host_id='assigned', method='PREPARATION:compile',
                       completion_sha256='accepted', detail={'outcome_status':'PREPARATION_FAILED'})
            marker = dict(job_id=job, request=request, source='fixed')
            C.put(root/(stage+'.DISTRIBUTED.json'), marker)
            result = root/'jobs'/row['method']/job
            C.put(result/'LAUNCH.json', dict(host_id='assigned', attempt_id='previous'))
            C.put(result/'PROCESS_RETURN.json', dict(returncode=2, signal=None))
            failure = dict(status='PREPARATION_FAILED', request_sha256=row['request_sha256'], artifacts={},
                           workspace_error=dict(step='git sparse-checkout add',returncode=2,scientific_body_started=False))
            C.put(result/'OUTCOME.json', failure)
            bridge = B.DistributedPreparationBridge(SimpleNamespace(run_dir=root,
                store=SimpleNamespace(row=lambda _:row), _user_stop_requested=lambda:False), root, approved=True)
            newer = deepcopy(request)
            newer.pop('request_id')
            newer['preparation']['inputs'][adapter] = 'a'*64
            with patch.object(bridge, 'build_request', return_value=newer):
                self.assertTrue(bridge.recoverable_transport(root, 'STAGE_FAILED:'+stage+':'+job))
                replacement = bridge.redeliver_pf_compile(None, [], root, stage, None, marker)
                self.assertEqual(replacement['request']['original_contract'], request['original_contract'])
                self.assertEqual(replacement['request']['preparation']['supersedes_preparation']['job_id'], job)
                failure['workspace_error']['scientific_body_started'] = True
                C.put(result/'OUTCOME.json', failure)
                self.assertIsNone(bridge.redeliver_pf_compile(None, [], root, stage, None, marker))

    def test_failed_preparation_recalculates_only_the_returned_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = normalize_preparation_request(dict(route=dict(kind='preparation'),
                original_contract={'calls': 2}, preparation=dict(stage='signal', provider='DF_SIGNALS',
                    source=dict(commit='fixed', files={}), command=['python', 'signals'],
                    outputs=['workspace/result'])))
            job = request['request_id']
            row = dict(request=request, request_sha256=W.digest(request), state='ACCEPTED',
                       attempt_id='failed', host_id='assigned', method='PREPARATION:signal',
                       completion_sha256='accepted')
            directory = root/'jobs'/row['method']/job
            C.put(directory/'REQUEST.json', request)
            C.put(directory/'LAUNCH.json', dict(attempt_id='failed', host_id='assigned'))
            C.put(directory/'PROCESS_RETURN.json', dict(returncode=2, signal=None, automatic_retry=False))
            outcome = dict(status='PREPARATION_FAILED', request_sha256=row['request_sha256'], artifacts={})
            C.put(directory/'OUTCOME.json', outcome)
            bridge = B.DistributedPreparationBridge(SimpleNamespace(
                run_dir=root, store=SimpleNamespace(row=lambda _:row)), root)
            plan = dict(attempt_id='failed', request_sha256=row['request_sha256'], reason='numerical gate')

            replacement = bridge.failed_preparation_request(job, plan)
            self.assertEqual(replacement, bridge.failed_preparation_request(job, plan))
            successor = replacement['preparation'].pop('supersedes_preparation')
            self.assertEqual(successor, dict(job_id=job, attempt_id='failed', reason='numerical gate'))
            self.assertEqual(normalize_preparation_request(replacement), request)
            C.put(directory/'OUTCOME.json', dict(outcome, artifacts={'workspace/result':'a'*64}))
            with self.assertRaisesRegex(ValueError, 'partial-result reconciliation'):
                bridge.failed_preparation_request(job, plan)
            row['state'] = 'UNKNOWN'
            with self.assertRaisesRegex(ValueError, 'terminal attempt'):
                bridge.failed_preparation_request(job, plan)

    def test_interrupted_calibration_preserves_fixed_inputs_and_only_reuses_scoped_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = normalize_preparation_request(dict(route=dict(kind='preparation'),
                original_contract={'calls': 4}, preparation=dict(stage='compile', provider='DF_CALIBRATION',
                    source=dict(commit='fixed', files={}), command=['python', 'acquire'],
                    working_input_hashes=[], outputs=['workspace/candidate'])))
            job = request['request_id']
            row = dict(request=request, request_sha256=W.digest(request), state='ACCEPTED',
                       attempt_id='interrupted', method='PREPARATION:compile', completion_sha256='accepted',
                       detail={'outcome_status': 'PREPARATION_FAILED'})
            job_path = root/'jobs'/row['method']/job
            C.put(job_path/'OUTCOME.json', dict(status='PREPARATION_FAILED', request_sha256=row['request_sha256']))
            C.put(job_path/'PREPARATION_RECEIPT.json', dict(returncode=None, signal=15, automatic_retry=False))
            cache = SimpleNamespace(missing=lambda _: [])
            bridge = B.DistributedPreparationBridge(SimpleNamespace(
                run_dir=root, store=SimpleNamespace(row=lambda _: row), artifact_cache=cache), root)
            path = 'workspace/candidate/df_prices/calibration/acquisition/m1/p0/axes_X/cosine.json'
            plan = dict(attempt_id='interrupted', request_sha256=row['request_sha256'],
                        reused_artifacts={path: 'a'*64})
            replacement = bridge.interrupted_calibration_request(job, plan)
            self.assertEqual(replacement['original_contract'], request['original_contract'])
            self.assertEqual(replacement['preparation']['command'], request['preparation']['command'])
            self.assertEqual(replacement['preparation']['source'], request['preparation']['source'])
            self.assertEqual(replacement['preparation']['inputs'], plan['reused_artifacts'])
            self.assertEqual(replacement['preparation']['working_input_hashes'], [dict(path=path, sha256='a'*64)])
            self.assertEqual(request['preparation']['inputs'], {})
            self.assertEqual(replacement, bridge.interrupted_calibration_request(job, plan))
            with self.assertRaisesRegex(ValueError, 'outside its original result scope'):
                bridge.interrupted_calibration_request(job, dict(plan, reused_artifacts={'workspace/replaced.py': 'b'*64}))
            row['state'] = 'UNKNOWN'
            with self.assertRaisesRegex(ValueError, 'terminal ownership'):
                bridge.interrupted_calibration_request(job, plan)

    def test_interrupted_calibration_automatically_verifies_returned_axes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = ArtifactCache(root/'artifacts')
            loc = dict(commit='a'*40, path='provider.py')
            context = dict(physical_candidate=dict(prefix=1), structure={}, binding=dict(rank=2),
                split=dict(lambda_R=1), one_body_policy='fixed', law={}, price_profile=dict(compiler='fixed'),
                implementation='fixed', expected_scalar=0, execution_source=loc, input_source=loc)
            physical_request = dict(physical_context=context, context_key=W.digest(context), m=0, axes=['X', 'Y'])
            physical_request['request_id'] = W.digest(physical_request)
            fixed_input = dict(output_directory='case/df_prices', cells=[dict(role='CALIBRATION', m=0,
                indices=[0], cohort='fixed', seeds=[[11]], request=physical_request)])
            input_bytes = json.dumps(fixed_input).encode()

            def put_blob(raw):
                sha = hashlib.sha256(raw).hexdigest()
                cache.put({sha:base64.b64encode(raw).decode()})
                return sha

            stage = 'DF_CALIBRATION'
            request = normalize_preparation_request(dict(route=dict(kind='preparation'),
                original_contract={'calls': 2}, preparation=dict(stage='compile', provider=stage,
                    source=dict(commit='fixed',files={}), controller_root=str(root),
                    command=['python','--inputs-commit',loc['commit']],
                    working_input_hashes=[], outputs=['workspace/case'],
                    inputs={'workspace/case/df_prices/INPUTS.json':put_blob(input_bytes)})))
            job = request['request_id']
            row = dict(job_id=job, request=request, request_sha256=W.digest(request), state='ACCEPTED',
                       attempt_id='original', host_id='assigned', method='PREPARATION:compile',
                       completion_sha256='accepted', detail={'outcome_status':'PREPARATION_FAILED'})
            result = root/'jobs'/row['method']/job
            marker = dict(job_id=job, request=request, source='fixed')
            C.put(root/(stage+'.DISTRIBUTED.json'), marker)
            C.put(result/'LAUNCH.json', dict(host_id='assigned', attempt_id='original'))
            receipt = dict(returncode=None, signal=15, automatic_retry=False)
            C.put(result/'PROCESS_RETURN.json', receipt)
            C.put(result/'PREPARATION_RECEIPT.json', dict(receipt, provider_session_drained=True))
            metrics = dict(rz_count=10, rz_depth=2, cx_count=0, cx_depth=0, total_depth=2, circuit_size=10)
            resolved = dict(deterministic_fragments=[0], randomized_fragments=[1], lambda_R=1,
                            scalar=0, preparation_hash='prep', tail_hash='tail')
            record = dict(request=physical_request, cohort='fixed', index=0, seeds=[11], event_digest='events',
                records=[dict(axis='X', C=10, S=6, D=4, metrics=metrics, fingerprint='wrapper')],
                source=dict(execution=loc, input=loc), resolved_context=resolved,
                resolved_context_key=W.digest(dict(context=context, resolved=resolved)))
            raw = dict(status='VALID', gate_status='INPUT_PREPARATION_REQUEST_LOWERING_COMPILER_FINGERPRINT_CHECKED',
                       m=0, index=0, record=record)
            path = 'workspace/case/df_prices/calibration/acquisition/m0/p0/axes_X/'
            artifacts = {path+'PATH_RECEIPT.json':put_blob(json.dumps(raw).encode()),
                         path+'cosine.json':put_blob(json.dumps(
                             dict(metrics=metrics, fingerprint='wrapper', compiler='fixed')).encode())}
            C.put(result/'OUTCOME.json', dict(status='PREPARATION_FAILED', request_sha256=row['request_sha256'],
                                            artifacts=artifacts))
            bridge = B.DistributedPreparationBridge(SimpleNamespace(
                run_dir=root, store=SimpleNamespace(row=lambda _:row), artifact_cache=cache,
                _user_stop_requested=lambda:False), root, approved=True)

            def fixed_bytes(argv, **kwargs):
                return input_bytes if argv[-1].endswith('INPUTS.json') else b'fixed provider'

            with patch.object(B.subprocess, 'check_output', side_effect=fixed_bytes):
                C.put(result/'PREPARATION_RECEIPT.json', receipt)
                self.assertIsNone(bridge._transport_repair(root, stage, marker))
                C.put(result/'PREPARATION_RECEIPT.json', dict(receipt, provider_session_drained=True))
                successor = bridge._transport_repair(root, stage, marker)['request']
                self.assertEqual(successor, bridge._transport_repair(root, stage, marker)['request'])
                self.assertEqual(successor['preparation']['inputs'], request['preparation']['inputs'] | artifacts)
                self.assertEqual(successor['original_contract'], request['original_contract'])
                raw['record']['seeds'] = [12]
                artifacts[path+'PATH_RECEIPT.json'] = put_blob(json.dumps(raw).encode())
                C.put(result/'OUTCOME.json', dict(status='PREPARATION_FAILED', request_sha256=row['request_sha256'],
                                                artifacts=artifacts))
                with self.assertRaisesRegex(ValueError, 'COHORT_INDEX_OR_SEED_MISMATCH'):
                    bridge._transport_repair(root, stage, marker)
                bridge.executor._user_stop_requested = lambda: True
                self.assertIsNone(bridge._transport_repair(root, stage, marker))

    def test_exec_rejection_repairs_only_unused_aliases_and_keeps_the_frozen_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = ArtifactCache(root/'artifacts')
            raw = b'{"raw":"/used/raw.json"}'
            sha = hashlib.sha256(raw).hexdigest()
            cache.path(sha).parent.mkdir(parents=True)
            cache.path(sha).write_bytes(raw)
            bindings = {'/fixed': '@SOURCE@', '/used': '@SOURCE@', '/unused': '@SOURCE@'}
            stage = 'DF_CALIBRATION'
            request = normalize_preparation_request(dict(route=dict(kind='preparation'),
                original_contract={'calls': 4}, preparation=dict(stage='compile', provider=stage,
                    source=dict(commit='fixed', files={}), command=['python', 'acquire'],
                    controller_root=str(root), input_hashes=[], inputs={'workspace/input.json': sha},
                    path_bindings=bindings)))
            job = request['request_id']
            row = dict(job_id=job, request=request, request_sha256=W.digest(request), state='ACCEPTED',
                       attempt_id='failed', method='PREPARATION:compile', completion_sha256='accepted',
                       detail={'outcome_status': 'PREPARATION_FAILED'})
            marker = dict(request=request, job_id=job, source='fixed')
            C.put(root/(stage+'.DISTRIBUTED.json'), marker)
            result_dir = root/'jobs'/row['method']/job
            C.put(result_dir/'PROCESS_RETURN.json', dict(returncode=2, signal=None))
            outcome = dict(status='PREPARATION_FAILED', request_sha256=row['request_sha256'], artifacts={},
                           workspace_error=dict(step='exec', error='E2BIG', scientific_body_started=False))
            C.put(result_dir/'OUTCOME.json', outcome)
            bridge = B.DistributedPreparationBridge(SimpleNamespace(
                run_dir=root, store=SimpleNamespace(row=lambda _: row), artifact_cache=cache), root)
            bridge.aliases = {'/used': '/fixed', '/unused': '/fixed'}
            replacement = bridge._transport_repair(root, stage, marker)['request']
            self.assertNotEqual(job, normalize_preparation_request(replacement)['request_id'])
            self.assertEqual(replacement['original_contract'], request['original_contract'])
            self.assertEqual(replacement['preparation']['inputs'], request['preparation']['inputs'])
            self.assertEqual(replacement['preparation']['path_bindings'], {'/fixed': '@SOURCE@', '/used': '@SOURCE@'})
            self.assertEqual(replacement['preparation']['supersedes_preparation']['attempt_id'], 'failed')
            self.assertEqual(request['preparation']['path_bindings'], bindings)
            C.put(root/'STOP.json', {})
            self.assertIsNone(bridge._transport_repair(root, stage, marker))
            (root/'STOP.json').unlink()
            outcome['workspace_error']['scientific_body_started'] = True
            C.put(result_dir/'OUTCOME.json', outcome)
            self.assertIsNone(bridge._transport_repair(root, stage, marker))

    def test_missing_received_artifact_resumes_the_original_stage_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = 'DF_CALIBRATION'
            job = 'prep-' + 'a'*24
            missing = root/'price-result.json'
            bridge = B.DistributedPreparationBridge.__new__(B.DistributedPreparationBridge)
            bridge.executor = SimpleNamespace(preparation_result=unittest.mock.Mock(
                return_value={'status': 'PREPARATION_COMPLETE'}))
            C.put(root/(stage+'.IMPORT_WAIT.json'), dict(job_id=job, missing_path=str(missing)))
            reason = 'PREPARATION_RESULT_ARTIFACTS_MISSING:'+stage+':'+job+':missing result'
            with patch.object(bridge, 'received_stage', return_value=True) as evidence:
                self.assertFalse(bridge.received_after_reservation_stop(root, reason))
                bridge.executor.preparation_result.assert_not_called()
                missing.write_text('{}')
                self.assertTrue(bridge.received_after_reservation_stop(root, reason))
                evidence.assert_called_with(root, stage, job)
                bridge.executor.preparation_result.assert_called_once_with(job, adopt=False)

    def test_unclaimed_stage_reopens_only_with_an_available_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = JobStore(root/'QUEUE.sqlite3')
            executor = SimpleNamespace(store=store, run_dir=root, state=SimpleNamespace(failure=None),
                                       _user_stop_requested=lambda: False)
            bridge = B.DistributedPreparationBridge(executor, root, {'budgets': {'compile': 0}}, approved=True)
            stage = 'DF_CALIBRATION'
            request = normalize_preparation_request(dict(route=dict(kind='preparation'), preparation=dict(
                stage='compile', provider=stage, artificial=True, source=dict(commit='fixed', files={}))))
            job = request['request_id']
            C.put(root/(stage+'.DISTRIBUTED.json'), dict(job_id=job, request=request, source='fixed'))
            hold = ('DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD:'+stage+':'+
                    str(dict(status='PREPARATION_BUDGET_OR_PERMISSION_HOLD', job_id=job)))
            self.assertFalse(bridge.resume_unreserved_stage(root, hold))
            bridge.settings['budgets']['compile'] = 1
            self.assertTrue(bridge.resume_unreserved_stage(root, hold))
            self.assertIsNone(store.row(job))
            self.assertTrue(store.reserve('PREPARATION:compile',job,request,1,True))
            store.claim('worker',1)
            self.assertIsNotNone(store.row(job)['attempt_id'])
            self.assertFalse(bridge.resume_unreserved_stage(root, hold))

    def test_import_received_stage_does_not_submit_a_new_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = B.DistributedPreparationBridge.__new__(B.DistributedPreparationBridge)
            outcome = dict(status='PREPARATION_COMPLETE')
            request = dict(preparation={'source': {'commit': 'FIXED'}})
            marker = root/'DF_CALIBRATION.DISTRIBUTED.json'
            C.put(marker, dict(job_id='original'))
            bridge.executor = SimpleNamespace(
                store=SimpleNamespace(row=lambda job: dict(request=request)),
                preparation_result=lambda job, **kwargs: outcome,
                submit_preparation=lambda *args: self.fail('Do not submit original stage again'),
                adopt_preparation=lambda job: self.assertEqual(job, 'original'))
            completion = SimpleNamespace(log=lambda **kwargs: None)
            with (patch.object(bridge,'received_stage',return_value=False),
                  patch.object(bridge,'_import_result') as restore):
                self.assertFalse(bridge.import_received_stage(completion, marker, 'DF_CALIBRATION'))
                restore.assert_not_called()
            with (patch.object(bridge,'received_stage',return_value=True),
                  patch.object(bridge,'_import_result') as restore):
                self.assertTrue(bridge.import_received_stage(completion, marker, 'DF_CALIBRATION'))
                restore.assert_called_once_with(outcome, request)
            self.assertEqual(C.read(root/'DF_CALIBRATION.RETURN.json')['job_id'],'original')

    def test_df_freeze_receipt_with_only_input_copies_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            method = 'PREPARATION:audit'
            prep = dict(provider='DF_MODEL_FREEZE', stage='audit', controller_root=str(root),
                        command=['provider', 'freeze', '--input', str(root/'candidate/df_prices/INPUTS.json')],
                        source={'commit': 'fixed', 'files': {}},
                        inputs={'workspace/candidate/df_prices/INPUTS.json': 'a'*64,
                                'workspace/candidate/df_prices/AXIS_PRICES.json': 'b'*64})
            request = normalize_preparation_request(
                dict(route={'kind': 'preparation'}, preparation=prep, original_contract={'source': 'fixed'}))
            job = request['request_id']
            candidate = root/'candidate'
            C.put(candidate/'DF_MODEL_FREEZE.DISTRIBUTED.json', dict(job_id=job, request=request, source='fixed'))
            executor = type('Engine', (), {'run_dir': root/'run', 'store': JobStore(root/'store.sqlite3')})()
            host = {'fixture': True}
            eligibility = dict(request_sha256=W.digest(request), eligible_host_ids=['fixed-host'],
                               host_fingerprints={'fixed-host': W.digest(host)})
            self.assertTrue(executor.store.reserve(method, job, request, 1, True, eligibility=eligibility))
            attempt = executor.store.claim('fixed-host', 1, host_fingerprints={job: host})[0]['attempt_id']
            executor.store.update('fixed-host', job, attempt, 'RUNNING')
            executor.store.update('fixed-host', job, attempt, 'COMPLETED_PENDING_TRANSFER',
                                  {'outcome_status': 'PREPARATION_COMPLETE'})
            executor.store.complete('fixed-host', job, attempt, W.digest(request), {})
            outcome = dict(status='PREPARATION_COMPLETE', request_sha256=W.digest(request),
                           artifacts=dict(prep['inputs'], **{
                               'workspace/other/df_prices/FROZEN_MODELS.json': 'c'*64,
                               'workspace/other/df_prices/AXIS_PRICES.json': 'd'*64,
                           }))
            result_path = executor.run_dir/'jobs'/method/job/'OUTCOME.json'
            C.put(result_path, outcome)
            executor.submit_preparation = lambda *a, **k: job
            executor.preparation_result = lambda *a, **k: outcome
            executor.adopt_preparation = executor.store.adopt
            completion = type('Completion', (), {'log': lambda *a, **k: None})()
            bridge = B.DistributedPreparationBridge(executor, root)
            with patch.object(bridge, '_import_result') as import_result:
                result = bridge.recovery_evidence(candidate, 'DF_MODEL_FREEZE', job)
                self.assertEqual(result['action'], 'CHECK_ORIGINAL_OUTPUT')
                self.assertEqual(result['attempt_id'], attempt)
                with self.assertRaisesRegex(C.Held, 'FROZEN_MODELS.json,AXIS_PRICES.json'):
                    bridge.call(completion, prep['command'], candidate, 'DF_MODEL_FREEZE')
                import_result.assert_not_called()
                self.assertIsNone(executor.store.row(job)['adopted_epoch'])
                self.assertFalse((candidate/'DF_MODEL_FREEZE.RETURN.json').exists())

                # A separate valid provider result would contain both outputs in
                # this exact candidate directory, with bytes absent from its inputs.
                outcome['artifacts'].update({
                    'workspace/candidate/df_prices/FROZEN_MODELS.json': 'e'*64,
                    'workspace/candidate/df_prices/AXIS_PRICES.json': 'f'*64,
                })
                C.put(result_path, outcome)
                self.assertEqual(bridge.recovery_evidence(candidate, 'DF_MODEL_FREEZE', job)['action'],
                                 'IMPORT_SUCCESSFUL_STAGE')
                bridge.call(completion, prep['command'], candidate, 'DF_MODEL_FREEZE')
                import_result.assert_called_once()
                self.assertIsNotNone(executor.store.row(job)['adopted_epoch'])

    def test_df_audit_requires_the_relocated_receipt_to_match_the_raw_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = 'search-evidence/old/completion/candidate/price/axes_X/cosine.json'
            current = 'workspace/search-evidence/new/completion/candidate/price/'
            path = root/old
            path.parent.mkdir(parents=True)
            path.write_bytes(b'{}\n')
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture', '-c',
                            'user.email=fixture@example.invalid', 'commit', '-qm', 'fixed'], check=True)
            commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            docs = [dict(commit='WORKING', path=old)]
            blobs = {current+'axes_X/cosine.json': sha}
            fixed, working = B.fixed_input_hashes(root, docs, blobs, df_raw_commit=(commit, current))
            self.assertEqual(fixed, [dict(commit=commit, path=old, sha256=sha)])
            self.assertEqual(working, [])
            blobs[current+'axes_X/cosine.json'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'Unfrozen or ambiguous WORKING'):
                B.fixed_input_hashes(root, docs, blobs, df_raw_commit=(commit, current))

    def test_leaf_keeps_only_its_used_worker_path_aliases(self):
        aliases = {'/old/a': '/fixed', '/old/b': '/fixed', '/old/c': '/fixed'}
        selected = B.used_path_aliases(aliases, {'/fixed': '@WORK@/input'},
            [{'path': '/old/a/receipt.json'}], ['--input', '/old/c'])
        self.assertEqual(selected, {'/old/a': '@WORK@/input', '/old/c': '@WORK@/input'})

    def test_df_signal_request_binds_selected_runner_without_legacy_bootstrap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def git(*args):
                return subprocess.check_output(
                    ['git', *args], cwd=root, text=True, stderr=subprocess.PIPE).strip()
            git('init')
            script = root / 'scripts/run_pr_df_candidate_signal_jobs.py'
            script.parent.mkdir()
            script.write_text('# artificial signal runner\n')
            git('add', '.')
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                'commit', '-qm', 'fixed artificial signal source')
            source = git('rev-parse', 'HEAD')
            work = root / 'candidate'
            work.mkdir()
            engine = type('Engine', (), {'run_dir': root / 'run'})()
            bridge = B.DistributedPreparationBridge(engine, root)
            completion = C.Completion.__new__(C.Completion)
            completion.source = 'legacy-source'
            completion.settings = {'df_signal_source': source, 'df_price_source': 'fixed-price'}
            command = [sys.executable, '-B', str(script), '--source', source]
            request = bridge.build_request(completion, command, work, 'DF_SIGNALS')
            self.assertEqual(request['preparation']['command'], command)
            self.assertEqual(request['original_contract']['source'], source)
            self.assertEqual(request['preparation']['source'], dict(
                commit=source, files={str(script.relative_to(root)):
                    hashlib.sha256(script.read_bytes()).hexdigest()}))
            self.assertEqual(completion.stage_source('DF_CALIBRATION'), 'fixed-price')
            self.assertEqual(completion.stage_source('PAULI_SIGNALS'), 'legacy-source')

    def test_native_runtime_paths_reuse_original_seals_and_ignore_optional_cache(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);provider=root/'provider';provider.mkdir();code=provider/'worker.py';code.write_text('# fixed native fixture\n')
            code_sha=hashlib.sha256(code.read_bytes()).hexdigest();config=root/'INPUT_CONFIG.json'
            C.put(config,{'sealed_files':[{'role':'provider/runtime','path':str(code),'sha256':code_sha},
                {'role':'unrelated-history','path':str(root/'absent'),'sha256':'0'*64}]})
            document={'runtime':{'provider_root':str(provider),'input_config':str(config),'old_cache':str(root/'optional-missing-cache')}}
            paths=B.collect_required_paths([document]);self.assertEqual(len(paths),2)
            directory=next(row for row in paths.values()if row['kind']=='directory')
            self.assertEqual(directory['files'],{'worker.py':code_sha})
            file=next(row for row in paths.values()if row['kind']=='file')
            self.assertEqual(file['sha256'],hashlib.sha256(config.read_bytes()).hexdigest())
            self.assertFalse(any('old_cache'in row['controller_path']for row in paths.values()))
            code.write_text('# changed source\n')
            with self.assertRaisesRegex(ValueError,'sealed provider/runtime file missing or changed'):
                B.collect_required_paths([document])
            config.unlink()
            with self.assertRaisesRegex(ValueError,'input_config missing'):
                B.collect_required_paths([document])

    def test_nested_git_locators_and_runtime_views_are_bound_without_import(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
            git('init');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
            payload=root/'CONFIGURATION.json';payload.write_text('{"selector": {"a": 1}, "preserve": true}\n')
            view_file=root/'src/trottertracks/pr_price_rules.py';view_file.parent.mkdir(parents=True);view_file.write_text('# metadata only, never imported\n')
            git('add','.');git('commit','-qm','fixed nested fixture');commit=git('rev-parse','HEAD')
            original={'nested':[{'input':{'commit':commit,'path':'CONFIGURATION.json','selector':['selector']}}],
                      'd2_view':str(root)}
            before=deepcopy(original)
            inputs,working=B.collect_fixed_inputs(root,[original])
            self.assertEqual(inputs,[{'commit':commit,'path':'CONFIGURATION.json','sha256':hashlib.sha256(payload.read_bytes()).hexdigest()}])
            self.assertEqual(working,[]);self.assertEqual(original,before)
            views=B.collect_required_views([original]);row=next(iter(views.values()))
            self.assertEqual(row['source']['commit'],commit);self.assertEqual(row['controller_path'],str(root))
            self.assertEqual(row['source']['files']['src/trottertracks/pr_price_rules.py'],hashlib.sha256(view_file.read_bytes()).hexdigest())
            view_file.write_text('# changed metadata SOURCE\n')
            with self.assertRaisesRegex(ValueError,'operational SOURCE view changed'):
                B.collect_required_views([original])

    def test_missing_fixed_input_metadata_holds_before_submission(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);engine=type('Engine',(),{'run_dir':root/'run'})()
            bridge=B.DistributedPreparationBridge(engine,root)
            completion=type('Completion',(),{'source':'fixed'})()
            with patch.object(bridge,'build_request',side_effect=ValueError('missing fixed input object')):
                with self.assertRaisesRegex(C.Held,'OPERATIONAL_HOLD.*INPUT_OR_SOURCE_METADATA_UNAVAILABLE'):
                    bridge.call(completion,['fixed'],root,'DF_SIGNALS')
            self.assertFalse((root/'DF_SIGNALS.DISTRIBUTED.json').exists())

    def test_prior_owner_markers_are_checked_at_original_path_before_submission(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);owner=root/'original-owner';output=owner/'docs/selected';output.mkdir(parents=True)
            directory=root/'new-controller';directory.mkdir()
            input_file=directory/'REQUEST.json';C.put(input_file,{'workspace':str(owner),'directory':'docs/selected'})
            marker=output/'SIGNALS_STARTED.json';C.put(marker,{'pid':123,'automatic_retry':False})
            engine=type('Engine',(),{'run_dir':root/'run'})()
            bridge=B.DistributedPreparationBridge(engine,root)
            completion=type('Completion',(),{'source':'fixed'})()
            with patch.object(bridge,'build_request',side_effect=AssertionError('must not clone/submit')):
                with self.assertRaisesRegex(C.Held,'ORIGINAL_OWNER_RUNNING_OR_UNKNOWN'):
                    bridge.call(completion,['fixed','--request',str(input_file)],directory,'PAULI_SIGNALS')
            self.assertFalse((directory/'PAULI_SIGNALS.DISTRIBUTED.json').exists())
            terminal=output/'run01/result.json';C.put(terminal,{'status':'ENTRY_CARRIERS_COMPLETE_NOT_RPE_QUALIFIED'})
            observation=bridge.inspect_prior_owner(['fixed',str(input_file)],directory,'PAULI_SIGNALS')
            self.assertFalse(observation['pending']);self.assertFalse(observation['failed'])
            C.put(output/'STOP.json',{'status':'PRESERVED_STOP'})
            observation=bridge.inspect_prior_owner(['fixed',str(input_file)],directory,'PAULI_SIGNALS')
            self.assertTrue(observation['failed'])
            self.assertTrue(marker.is_file());self.assertEqual(json.loads(input_file.read_text())['workspace'],str(owner))

    def test_operational_hold_reconciles_same_attempt_but_failed_science_does_not_retry(self):
        for first_status in ('HOST_ELIGIBILITY_HOLD', 'PREPARATION_FAILED', 'UNKNOWN', 'PREPARATION_STOPPED_BEFORE_RESERVATION'):
            with self.subTest(first_status=first_status),tempfile.TemporaryDirectory()as tmp:
                root=Path(tmp)
                class Engine:
                    run_dir=root/'run'
                    state=SimpleNamespace(failure=None)
                    def _user_stop_requested(self):return False
                    def __init__(self):self.store=JobStore(root/'store.sqlite3');self.requests=[];self.result_calls=0;self.starts=0
                    def submit_preparation(self,request,**kwargs):
                        request = normalize_preparation_request(request)
                        self.requests.append(request)
                        job = request['request_id']
                        self.store.reserve('PREPARATION:signal',job,request,1,True)
                        if self.store.row(job)['state']=='QUEUED':
                            row=self.store.claim('original-host',1)[0];self.attempt=row['attempt_id'];self.starts+=1
                            self.store.update('original-host',job,self.attempt,'RUNNING')
                        return job
                    def preparation_result(self,job,**kwargs):
                        self.result_calls+=1
                        if self.result_calls==1:
                            if first_status=='UNKNOWN':self.store.update('original-host',job,self.attempt,'UNKNOWN')
                            return {'status':first_status, 'job_id':job}
                        self.store.complete('original-host',job,self.attempt,W.digest(self.requests[-1]),{})
                        return {'status':'PREPARATION_COMPLETE','artifact_directory':str(root),'artifacts':{}}
                    def adopt_preparation(self, job):
                        self.store.adopt(job)
                class Fake(C.Completion):
                    def pauli_configuration(self,request,d,allow_science=True):return request['configuration'],{}
                    def initial_quote(self,*args):return {'complete':True,'rows':[{'unit_price_RZ':1}]}
                    def pauli_signals(self,config,prepared,d):
                        self.call(['fixed'],d,'PAULI_SIGNALS',science=True)
                        return {'packet_source':{'commit':'saved','path':'PACKET'}}
                    def price(self,config,cref,prepared,r,d,initial_quote=None):
                        q=self.initial_quote();C.put(d/'FINAL_QUOTE.json',q);return q
                    def service(self,mode,*args,**kwargs):return {'binding_id':'binding','total_configuration_id':'total'}
                c=Fake.__new__(Fake);c.directory=root;c.settings={};c.source='fixed';c.lock=threading.RLock()
                c.shared_preparation_locks={};c.state={'records':{}};c.state_path=root/'STATE.json';c.registry=root/'REGISTRY.json'
                C.put(c.registry,{'entries':[]})
                ex=Engine();bridge=B.DistributedPreparationBridge(ex,root,{'budgets':{'signal':1}},approved=True);c.bind_stage_executor(bridge)
                frozen={'route':{'kind':'preparation'},'preparation':{'artificial':True,'stage':'signal',
                        'provider':'PAULI_SIGNALS','source':{'commit':'fixed','files':{}}},
                        'original_contract':{'source':'fixed'}}
                request={'method':'PR_PAULI','configuration':{'id':'a','spec':{'L_D':4,'formula':'2nd','kappa':{'value':'2'},'delta_pf':'1'},'rounds':[]}}
                with patch.object(C,'ROOT',root),patch.object(C,'publish',return_value='saved'),\
                     patch.object(C,'value',side_effect=lambda loc:C.read(root/loc['path'])),\
                     patch.object(bridge,'build_request',return_value=frozen)as builder,\
                     patch.object(bridge,'_import_result'):
                    first=c.complete(request)
                    job=ex.requests[0]['request_id']
                    attempt=ex.store.row(job)['attempt_id']
                    if first_status == 'PREPARATION_STOPPED_BEFORE_RESERVATION':
                        self.assertEqual(first['status'], 'HELD_WITH_REASON')
                        self.assertEqual(c.complete(request)['status'], 'HELD_WITH_REASON')
                        self.assertEqual(len(ex.requests), 1)
                        ex.store.update('original-host',job,attempt,'COMPLETED_PENDING_TRANSFER',
                                        {'outcome_status':'PREPARATION_COMPLETE'})
                        ex.store.complete('original-host', job, attempt, W.digest(ex.requests[0]), {})
                        directory = root/'candidates'/first['key']
                        marker = directory/'PAULI_SIGNALS.DISTRIBUTED.json'
                        saved = C.read(marker)
                        changed = deepcopy(saved)
                        changed['request']['original_contract']['source'] = 'changed'
                        C.put(marker, changed)
                        self.assertFalse(bridge.received_after_reservation_stop(directory, first['reason']))
                        C.put(marker, saved)
                    second=c.complete(request)
                self.assertEqual(ex.starts,1);self.assertEqual(ex.store.row(job)['attempt_id'],attempt)
                self.assertEqual(builder.call_count,1)
                if first_status != 'PREPARATION_FAILED':
                    if first_status != 'PREPARATION_STOPPED_BEFORE_RESERVATION':
                        self.assertEqual(first['status'],'WAITING_DISTRIBUTED_PREPARATION')
                    self.assertEqual(second['status'],'READY_FOR_SEARCH')
                    self.assertEqual(len(ex.requests),2);self.assertEqual(ex.requests[0],ex.requests[1])
                    self.assertEqual(len([e for e in ex.store.events()if e['event']=='ADOPT']),1)
                else:
                    self.assertTrue(first['scientific_failure']);self.assertTrue(second['resumed'])
                    self.assertEqual(len(ex.requests),1)

    def test_runtime_requirements_follow_original_input_and_fixed_worker_guard(self):
        with tempfile.TemporaryDirectory()as tmp:
            script=Path(tmp)/'fixed.py';script.write_text("if (numpy.__version__, scipy.__version__) != ('1.26.4','1.14.1'): raise ValueError('runtime')\n")
            inputs=[{'configuration':{'profile':{'qiskit_version':'1.3.0','compiler':'a'*64}}}]
            requirements,bindings=B.contract_runtime(inputs,script,{})
            self.assertEqual(requirements['packages'],{'qiskit':'1.3.0','numpy':'1.26.4','scipy':'1.14.1'})
            self.assertEqual(bindings,['a'*64])
            with self.assertRaisesRegex(ValueError,'conflicts with original input'):
                B.contract_runtime(inputs,script,{'packages':{'qiskit':'2.0.0'}})

    def test_pf_wrapper_carries_distinct_provider_manifest_and_rejects_change(self):
        with tempfile.TemporaryDirectory()as tmp:
            top=Path(tmp);transport=top/'transport';transport.mkdir();provider=top/'provider';provider.mkdir()
            def git(*args):
                return subprocess.check_output(['git',*args],cwd=provider,text=True,stderr=subprocess.DEVNULL).strip()
            git('init');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
            files={}
            for index in range(5):
                name='scripts/fixed_provider_'+str(index)+'.py';path=provider/name;path.parent.mkdir(exist_ok=True)
                path.write_text('# fixed provider fixture '+str(index)+'\n');files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            for filename in ('references/base_files_v1.json', '.github/workflows/validation-manifest.yml'):
                support = provider/filename
                support.parent.mkdir(parents=True, exist_ok=True)
                support.write_text('fixture: '+filename+'\n')
            (provider/'requirements.txt').write_text('openfermionpyscf==0.5\nopenfermion==1.6.1\npyscf==2.7.0\n')
            git('add','.');git('commit','-qm','fixed PF fixture');source=git('rev-parse','HEAD')
            wrapper=transport/'wrapper.py';wrapper.write_text('# transport adapter fixture\n')
            d=provider/'docs/owned';d.mkdir(parents=True);C.put(d/'REQUEST.json',{'scope':'fixture'})
            executor=type('Executor',(),{'run_dir':top/'run','artifact_cache':ArtifactCache(top/'cas')})()
            bridge=B.DistributedPreparationBridge(executor,transport)
            completion=type('Completion',(),{'source':source,'preparation_root':provider,
                'preparation_source_manifest':{'commit':source,'files':files}})()
            task=bridge.build_request(completion,[sys.executable,'-B',str(wrapper),'--input',str(d/'REQUEST.json')],d,'PF_COMPILE')
            self.assertEqual(task['preparation']['runtime_requirements']['packages'],
                             {'openfermionpyscf': '0.5', 'openfermion': '1.6.1', 'pyscf': '2.7.0'})
            self.assertEqual(task['preparation']['source'],{'commit':source,'files':files})
            self.assertEqual(task['preparation']['path_bindings'][str(transport)],'@TRANSPORT@')
            support_bundle, = [x for x in task['preparation']['input_git_bundles'] if x.get('workspace_objects')]
            holdout = bridge.build_request(completion,[sys.executable,'-B',str(wrapper),'--input',str(d/'REQUEST.json')],d,'PF_HOLDOUT_COMPILE')
            self.assertIn(support_bundle, holdout['preparation']['input_git_bundles'])
            sparse = top/'sparse'
            subprocess.run(['git','clone','-q','--no-hardlinks',str(provider),str(sparse)],check=True)
            missing = []
            for filename in ('references/base_files_v1.json', '.github/workflows/validation-manifest.yml'):
                oid = git('rev-parse',source+':'+filename)
                (sparse/'.git/objects'/oid[:2]/oid[2:]).unlink()
                missing.append(oid)
            subprocess.run(['git','-C',str(sparse),'bundle','unbundle',
                            str(executor.artifact_cache.path(support_bundle['sha256']))],check=True,capture_output=True)
            for oid in missing:
                subprocess.run(['git','-C',str(sparse),'cat-file','-e',oid],check=True)
            task['preparation']['artificial']=True;task['request_id']='pf-fixture'
            assignment={'job_id':'pf-fixture','request':task,'request_sha256':W.digest(task),'source_manifest':task['preparation']['source']}
            profile={'host_id':'fixture','root':str(provider),'interpreter':sys.executable,'work_dir':str(top/'spool')}
            W.worker_cache(profile).put(executor.artifact_cache.get([support_bundle['sha256']]))
            W.current_preparation_fingerprint(assignment,profile)
            (provider/next(iter(files))).write_text('# changed source\n')
            with self.assertRaisesRegex(ValueError,'SOURCE hash differs'):
                W.current_preparation_fingerprint(assignment,profile)

    def test_new_locator_bundle_reaches_distinct_owned_provider_workspace(self):
        with tempfile.TemporaryDirectory()as tmp:
            top=Path(tmp);root=top/'controller';root.mkdir()
            def git(*args,cwd=root):
                return subprocess.check_output(['git',*args],cwd=cwd,text=True,stderr=subprocess.DEVNULL).strip()
            git('init');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
            helper=root/'src/trottertracks/rpe_preparation_paths.py';helper.parent.mkdir(parents=True)
            shutil.copy2(ROOT/'src/trottertracks/rpe_preparation_paths.py',helper)
            fixed_provider=root/'scripts/run_pauli_split_cal.py';fixed_provider.parent.mkdir()
            fixed_provider.write_text('# artificial source guard only\n')
            script=root/'scripts/artificial_stage.py'
            script.write_text('''import hashlib,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks.rpe_preparation_paths import resolve
raw=Path(sys.argv[1]).read_bytes();request=json.loads(raw)
workspace=Path(resolve(request['workspace']))
config=json.loads(subprocess.check_output(['git','show',request['configuration']['commit']+':'+request['configuration']['path']],cwd=workspace))
out=workspace/request['directory'];out.mkdir(parents=True,exist_ok=True)
(out/'TOUCHED.json').write_text(json.dumps({'fixture':True}))
subprocess.run(['git','add',str(out/'TOUCHED.json')],cwd=workspace,check=True)
subprocess.run(['git','commit','-qm','artificial provider receipt'],cwd=workspace,check=True)
publication=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()
Path(sys.argv[2]).write_text(json.dumps({'configuration':config,'input_sha256':hashlib.sha256(raw).hexdigest(),'workspace':str(workspace),'publication':publication}))
''')
            historic=root/'old'/'CONFIGURATION.json';historic.parent.mkdir()
            historic.write_text('{"previous_proof": 17}\n')
            git('add','.');git('commit','-qm','historical fixed input');historic_commit=git('rev-parse','HEAD')
            historic_blob=git('rev-parse',historic_commit+':old/CONFIGURATION.json')
            historic.unlink()
            git('add','.');git('commit','-qm','fixed artificial provider');source=git('rev-parse','HEAD')
            provider=top/'original-provider';git('clone','-q',str(root),str(provider))
            # A real remote host has a separate Git object database. Clone
            # before the controller creates its unpublished input commits.
            view=top/'fixed-view';git('clone','-q',str(root),str(view))
            # A source checkout can resolve the historical commit and lack the
            # particular historical object. The new bundle must supply it.
            (view/'.git'/'objects'/historic_blob[:2]/historic_blob[2:]).unlink()
            config=root/'docs/CONFIGURATION.json';config.parent.mkdir();config.write_text('{"scientific_fixture": "immutable"}\n')
            git('add','.');git('commit','-qm','new controller input');config_commit=git('rev-parse','HEAD')
            d=root/'docs/candidate';d.mkdir()
            request=d/'REQUEST.json';request.write_text(json.dumps({'workspace':str(provider),'directory':'docs/provider-output',
                'configuration':{'commit':config_commit,'path':'docs/CONFIGURATION.json'},
                'historical':{'commit':historic_commit,'path':'old/CONFIGURATION.json'}})+'\n')
            pinned_helper=hashlib.sha256(helper.read_bytes()).hexdigest()
            helper.write_bytes(helper.read_bytes()+b'\n# New controller transport declaration.\n')
            (root/'shared-history.bin').write_bytes(Random(20261118).randbytes(1024 * 1024))
            git('add','.');git('commit','-qm','frozen task input')
            executor=type('Executor',(),{'run_dir':top/'run','artifact_cache':ArtifactCache(top/'controller-cas')})()
            git('checkout', '-b', 'earlier-result', source)
            (root/'unrelated.json').write_text('{"another_candidate": true}\n')
            git('add', 'unrelated.json');git('commit', '-qm', 'unrelated producer result')
            unrelated = git('rev-parse', 'HEAD')
            earlier = 'refs/rpe-preparation/'+B.digest(str(executor.run_dir))[:16]+'/earlier-result'
            git('update-ref', earlier, unrelated)
            git('checkout', '-')
            bridge=B.DistributedPreparationBridge(executor,root,{'bundle_validation_sources':{source:str(view)}})
            completion=type('Completion',(),{'source':source})()
            task=bridge.build_request(completion,[sys.executable,'-B',str(script),str(request),str(d/'RESULT.json')],d,'PAULI_SIGNALS')
            bundles = task['preparation']['input_git_bundles']
            base, inputs, workspace = bundles
            self.assertTrue(workspace['workspace_objects'])
            self.assertFalse(workspace['checkout_input_head'])
            self.assertGreater(executor.artifact_cache.path(base['sha256']).stat().st_size, 1024 * 1024)
            self.assertLess(executor.artifact_cache.path(inputs['sha256']).stat().st_size, 64 * 1024)
            other_run = type('Executor', (), {'run_dir': top/'other-run', 'artifact_cache': executor.artifact_cache})()
            other_bridge = B.DistributedPreparationBridge(other_run, root)
            locators = task['preparation']['input_hashes']
            self.assertEqual(other_bridge._bundle(root, source, locators), bundles[:2])
            tracked_base, tracked = bridge._bundle(root, source, locators[1:], tracked_workspace=True)
            self.assertEqual(tracked_base, base)
            self.assertNotEqual(tracked['sha256'], inputs['sha256'])
            self.assertLess(executor.artifact_cache.path(tracked['sha256']).stat().st_size, 64 * 1024)
            self.assertTrue(tracked['checkout_input_head'])
            tracked_view = W._clone(view, top/'tracked-view', [tracked_base, tracked], executor.artifact_cache)
            self.assertEqual(git('rev-parse', 'HEAD^{tree}', cwd=tracked_view), git('rev-parse', 'HEAD^{tree}'))
            start = git('rev-parse', 'HEAD', cwd=tracked_view)
            return_path = top/'delta.bundle'
            fixed_view = dict(source=dict(commit=source), input_git_bundles=[tracked_base, tracked])
            receiver = top/'clean-receiver'
            git('clone', '--shared', '-q', str(view), str(receiver))
            returning = dict(preparation=dict(fixed_view,controller_root=str(receiver)))
            unchanged = W._export_bundle(tracked_view, fixed_view['source'], return_path, start)
            self.assertNotIn('path', unchanged)
            bridge._import_result(dict(git_bundles=[unchanged],artifact_directory=str(top)), returning)
            self.assertEqual(git('cat-file', '-t', start, cwd=receiver), 'commit')
            self.assertEqual(git('rev-parse', 'HEAD', cwd=receiver), source)
            (tracked_view/'new-result.json').write_text('{"fixture":42}\n')
            git('add', 'new-result.json', cwd=tracked_view)
            git('commit', '-qm', 'one new result', cwd=tracked_view)
            delta = W._export_bundle(tracked_view, fixed_view['source'], return_path, start)
            self.assertLess(return_path.stat().st_size, 64*1024)
            outcome = dict(git_bundles=[delta],artifact_directory=str(top),
                           artifacts={return_path.name:hashlib.sha256(return_path.read_bytes()).hexdigest()})
            bridge._import_result(outcome, returning)
            bridge._import_result(outcome, returning)
            self.assertEqual(git('show', delta['head']+':new-result.json', cwd=receiver), '{"fixture":42}')
            with self.assertRaisesRegex(ValueError, 'assigned input HEAD'):
                bridge._import_result(dict(outcome,git_bundles=[dict(delta,input_head=source)]), returning)
            self.assertEqual(task['preparation']['workspaces']['provider']['source']['files'][
                'src/trottertracks/rpe_preparation_paths.py'],pinned_helper)
            fixed_provider.write_text('# changed provider body\n')
            with self.assertRaisesRegex(ValueError,'Provider workspace SOURCE differs'):
                bridge.build_request(completion,[sys.executable,'-B',str(script),str(request),str(d/'RESULT.json')],d,'PAULI_SIGNALS')
            fixed_provider.write_text('# artificial source guard only\n')
            root_hashes={v['sha256']for v in task['preparation']['input_git_bundles']}
            provider_hashes={v['sha256']for v in task['preparation']['workspaces']['provider']['input_git_bundles']}
            self.assertTrue({base['sha256'], inputs['sha256']} <= provider_hashes)
            task['preparation']['artificial']=True;task['request_id']='fixture'
            assignment={'job_id':'fixture','request':task,'request_sha256':W.digest(task),'source_manifest':task['preparation']['source']}
            profile={'host_id':'simulated-other-host','root':str(view),'interpreter':sys.executable,
                     'work_dir':str(top/'worker-spool'),'preparation_sources':{source:str(view)},
                     'artifact_cache_directory':str(top/'worker-cas')}
            hashes=set(task['preparation']['inputs'].values())|root_hashes|provider_hashes
            worker_cache=ArtifactCache(top/'worker-cas')
            worker_cache.put(executor.artifact_cache.get(hashes))
            self.assertNotEqual(subprocess.run(['git','cat-file','-e',config_commit],cwd=view,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode,0)

            # The legacy clone imports the same pack into each workspace.
            legacy = W._clone(view, top/'legacy-worker-view', bundles, worker_cache)
            legacy_packs = list((legacy/'.git/objects/pack').glob('*.pack'))
            self.assertGreater(sum(path.stat().st_size for path in legacy_packs), 1024 * 1024)
            attempt=top/'worker-attempt';outcome=W.execute(assignment,profile,attempt)
            self.assertEqual(outcome['status'],'PREPARATION_COMPLETE')
            self.assertFalse((attempt/'inputs').exists())
            self.assertEqual(W.collect(assignment,profile,attempt), outcome)
            shared = W._metadata_repository_path(profile, view)/'objects'
            for workspace in (attempt/'workspace', attempt/'workspaces/provider'):
                objects = workspace/'.git/objects'
                self.assertIn(str(shared), (objects/'info/alternates').read_text().splitlines())
                self.assertFalse(list((objects/'pack').glob('*.pack')))
            result=json.loads((attempt/'workspace/docs/candidate/RESULT.json').read_text())
            self.assertEqual(result['configuration'],{'scientific_fixture':'immutable'})
            self.assertEqual(result['input_sha256'],hashlib.sha256(request.read_bytes()).hexdigest())
            self.assertEqual(git('show',historic_commit+':old/CONFIGURATION.json',cwd=attempt/'workspace'),'{"previous_proof": 17}')
            unexpected = subprocess.run(['git', 'cat-file', '-e', unrelated], cwd=attempt/'workspace', capture_output=True)
            self.assertNotEqual(unexpected.returncode, 0)
            self.assertNotEqual(result['workspace'],str(provider))
            self.assertEqual(git('rev-parse','HEAD',cwd=view),source)
            self.assertFalse((provider/'docs/provider-output').exists())
            original_head=git('rev-parse','HEAD')
            bridge._import_result(dict(outcome,artifact_directory=str(attempt)),task)
            published=json.loads(git('show',result['publication']+':docs/provider-output/TOUCHED.json'))
            self.assertEqual(published,{'fixture':True})
            self.assertEqual(git('rev-parse','HEAD'),original_head)
            self.assertEqual(request.read_bytes(),(attempt/'workspace/docs/candidate/REQUEST.json').read_bytes())
            # A returned --output file describes the previous execution host.
            # It is dependency evidence, not the next task's runtime request.
            returned=d/'RESULT.json';returned.write_text(json.dumps(dict(result,workspace='/not-on-controller/old-attempt')))
            next_task=bridge.build_request(completion,[sys.executable,'-B',str(script),str(request),'--output',str(returned)],d,'PAULI_AUDIT')
            self.assertNotIn('/not-on-controller/old-attempt',next_task['preparation']['path_bindings'])
            self.assertIn(str(provider),next_task['preparation']['path_bindings'])

    def test_incomplete_descriptor_locator_still_holds_before_signal(self):
        c=C.Completion.__new__(C.Completion);c.settings={};c.state={'records':{}}
        config={'id':'fixture','spec':{'L_D':4,'binding':{'H':'fixed'}}}
        prepared={'descriptor':{'source':{'commit':'fixed'},'descriptor':{'tail':'own'}}}
        with patch.object(c,'call',side_effect=AssertionError('must not start signal')):
            with self.assertRaisesRegex(C.Held,'PAULI_DESCRIPTOR_SOURCE_UNRESOLVED'):
                c.pauli_signals(config,prepared,Path('/unused'))

    def test_two_declared_candidates_share_inflight_physical_work(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            class Fake(C.Completion):
                def pauli_configuration(self,request,d,allow_science=True):return request['configuration'],{}
                def initial_quote(self,*args):return {'complete':True,'rows':[{'unit_price_RZ':1}]}
                def pauli_signals(self,config,prepared,d):
                    with self.lock:self.signal_count+=1
                    self.branch_barrier.wait(timeout=2)
                    return {'packet_source':{'commit':'saved','path':'PACKET'}}
                def price(self,config,cref,prepared,r,d,initial_quote=None):
                    with self.lock:self.compile_count+=1
                    self.branch_barrier.wait(timeout=2)
                    q=self.initial_quote();C.put(d/'FINAL_QUOTE.json',q);return q
                def service(self,mode,*args,**kwargs):
                    if mode=='intake':return {'binding_id':'binding','total_configuration_id':'total'}
                    raise AssertionError(mode)
            c=Fake.__new__(Fake);c.directory=root;c.settings={};c.source='fixed'
            c.lock=threading.RLock();c.shared_preparation_locks={};c.stage_executor=object()
            c.state={'records':{}};c.state_path=root/'STATE.json';c.registry=root/'REGISTRY.json'
            C.put(c.registry,{'entries':[]});c.branch_barrier=threading.Barrier(2)
            c.signal_count=c.compile_count=0
            cfg={'id':'a','spec':{'L_D':4,'formula':'2nd','kappa':{'value':'2'},'delta_pf':'1',
                'binding':{'H':'fixed'},'split':{'dictionary':'fixed'}},'rounds':[]}
            other=deepcopy(cfg);other['id']='b';other['spec']['kappa']['value']='3'
            with patch.object(C,'ROOT',root),patch.object(C,'publish',return_value='saved'),\
                 patch.object(C,'value',side_effect=lambda loc:C.read(root/loc['path'])):
                with ThreadPoolExecutor(max_workers=2)as pool:
                    results=list(pool.map(c.complete,[{'method':'PR_PAULI','configuration':cfg},
                                                      {'method':'PR_PAULI','configuration':other}]))
            self.assertEqual([r['status']for r in results],['READY_FOR_SEARCH']*2)
            self.assertEqual((c.signal_count,c.compile_count),(1,1))
            self.assertEqual(sum('reused_physical_configuration'in r for r in results),1)

    def test_heavy_call_uses_queue_recovery_before_started_marker(self):
        with tempfile.TemporaryDirectory()as tmp:
            c=C.Completion.__new__(C.Completion)
            runtime=type('Runtime',(),{'handles':staticmethod(lambda stage:True),
                'call':staticmethod(lambda *args,**kwargs:{'status':'RECONCILED'})})()
            c.stage_executor=runtime
            C.put(Path(tmp)/'DF_SIGNALS.STARTED.json',{'pid':123})
            self.assertEqual(c.call(['fixed'],Path(tmp),'DF_SIGNALS',science=True),{'status':'RECONCILED'})

    def test_budget_default_is_hold_and_no_result_import(self):
        with tempfile.TemporaryDirectory()as tmp:
            class Executor:
                run_dir=tmp
                def submit_preparation(self,request,**kwargs):self.args=kwargs;return 'task'
                def preparation_result(self,*args,**kwargs):return {'status':'PREPARATION_BUDGET_OR_PERMISSION_HOLD'}
            ex=Executor();bridge=B.DistributedPreparationBridge(ex,tmp,approved=True)
            completion=type('C',(),{'source':'fixed'})()
            frozen={'preparation':{'source':{'commit':'fixed'}},'original_contract':{}}
            with patch.object(bridge,'build_request',return_value=frozen):
                with self.assertRaisesRegex(C.Held,'BUDGET_OR_PERMISSION_HOLD'):
                    bridge.call(completion,[],tmp,'DF_SIGNALS')
            self.assertEqual(ex.args['budget'],0)

    def test_pauli_audit_never_acquires_missing_compile(self):
        path=ROOT/'scripts/run_pr_pauli_candidate_prices.py'
        spec=importlib.util.spec_from_file_location('_prep_pauli_phase',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            for name in module.A_FILES+module.AUDIT_FILES:
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b'')
            cfg={'id':'fixture','spec':{'L_D':4}}
            request={'workspace':tmp,'directory':'owned','configuration':cfg,
                     'price_quote':{'configuration_id':'fixture','policy':'MODEL','complete':False},
                     'prepared_item':{'configuration':cfg}}
            with patch.object(module.OPS,'owned_checkout_lock',return_value=nullcontext()),\
                 patch.object(module.subprocess,'check_output',return_value=b''),\
                 patch.object(module,'command',side_effect=AssertionError('audit must not acquire')):
                result=module._run_one(request,execute=True,push=False,phase='audit')
            self.assertEqual(result['status'],'HELD_AUDIT_COMPILE_DEPENDENCY_MISSING')
            self.assertEqual(result['new_compiles_this_invocation'],0)
        self.assertIn('PF_HOLDOUT_COMPILE',B.STAGES)
        self.assertIn('PF_HOLDOUT_AUDIT',B.STAGES)


if __name__=='__main__':unittest.main()
