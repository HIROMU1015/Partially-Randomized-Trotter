"""Verified WSL cells through artificial claims and real worker command assembly.

Baseline three-host checks are reused, not rerun. Only currently PASS WSL cells
are exercised; remote metadata and every process launch are evidence stubs.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_q3_coordinator as Q
from trottertracks import rpe_q3_worker as W
from trottertracks.rpe_host_profiles import profile_core
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor

DOC = ROOT / 'docs/research/rpe_q3_host_eligibility'
WSL_DOC = DOC / 'optional_wsl'
BASE = ROOT / 'docs/research/rpe_distributed_h4_plan_20261117'
HOST = 'local0-wsl'


def read(path):
    return json.loads(Path(path).read_text())


class WslProfileAssignmentTests(unittest.TestCase):
    def test_verified_wsl_pass_cells_preserve_selected_profile_and_owned_attempt(self):
        policy_path = WSL_DOC / 'HOST_ELIGIBILITY.json'
        if not policy_path.exists(): policy_path = WSL_DOC / 'INITIAL_HOST_ELIGIBILITY.json'
        policy = read(policy_path)
        eligible = {job: row for job,row in policy['jobs'].items()
                    if row['hosts'].get(HOST, {}).get('status') == 'PASS'}
        self.assertTrue(eligible, 'At least one verified WSL cell is needed')
        candidates = [read(p) for p in sorted((BASE / 'assignments').glob('*.json'))]
        candidates.append(read(DOC / 'pauli_descriptor_route/ASSIGNMENT.json'))
        assignments = {}
        for job, decision in eligible.items():
            matches = [a for a in candidates if a['job_id'] == job and
                       Q.digest(a['request']) == decision['request_sha256']]
            self.assertTrue(matches, 'No matching public scientific REQUEST: ' + decision['method'])
            assignments[job] = matches[0]
            self.assertIn(HOST, decision['eligible_host_ids'])
            self.assertEqual(matches[0]['source_manifest'], decision['source_manifest'])
        first = next(iter(policy['jobs'].values()))
        computation = Path(os.environ.get('RPE_WSL_COMPUTATION_ROOT',
            first['hosts']['server2']['execution_profile']['root']))
        if not (computation / '.git').exists():
            self.skipTest('Needs the existing fixed computation checkout')
        manifest = first['source_manifest'];source = manifest['commit']
        self.assertEqual(Q.source_manifest(computation, source), manifest)
        original_bytes = {name:(computation / name).read_bytes() for name in manifest['files']}
        input_blobs = {}
        for assignment in assignments.values():
            for locator in assignment['input_hashes']:
                key = locator['commit'] + ':' + locator['path']
                if key not in input_blobs:
                    input_blobs[key] = subprocess.check_output(['git','show',key],cwd=computation)
                self.assertEqual(hashlib.sha256(input_blobs[key]).hexdigest(),locator['sha256'])
        problem = read(BASE / 'inputs/PROBLEM.json')
        controller_source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        report_rows = []
        with tempfile.TemporaryDirectory(prefix='rpe-artificial-wsl-profile-') as temporary:
            for job, decision in eligible.items():
                with self.subTest(method=decision['method']):
                    directory = Path(temporary) / job;directory.mkdir()
                    run = directory / 'ARTIFICIAL-controller'
                    cell = decision['hosts'][HOST];want = cell['execution_profile']
                    assignment = assignments[job]
                    routes = {assignment['method']:[assignment['request']['route']]}
                    expected = {assignment['request']['configuration_id']:assignment['request']['expected']}
                    options = dict(bind='127.0.0.1',port=0,affinity_wait_seconds=0,
                        computation_root=str(computation),computation_source=source,
                        host_eligibility=policy,host_admission=dict(schema='RPE_Q3_HOST_ADMISSION_V1',
                            hosts={HOST:dict(accepting=False,optional=True,max_slots=1)}))
                    def create():
                        return DistributedExecutor(ROOT,run,routes,expected,problem,optimizer_source=controller_source,
                            max_concurrent=4,budgets={assignment['method']:1},science_approved=True,distributed=options)
                    def release(ex):
                        if ex.state.driver.closed:return
                        with ex.state.cv:
                            for task in ex.state.tasks.values():
                                task.status='DONE';task.error='ARTIFICIAL_NO_OUTCOME';task.event.set()
                            ex.state.active=0;ex.state.cv.notify_all()
                        ex.finish()
                    executor = create()
                    try:
                        allow = executor._eligibility_for(assignment['request'],job)
                        self.assertIn(HOST,allow['eligible_host_ids'])
                        self.assertTrue(executor.store.reserve(assignment['method'],job,assignment['request'],1,True,
                            contract={'fixture':'ARTIFICIAL_METADATA_QUEUE_ONLY'},eligibility=allow))
                        with executor.state.cv:executor._restore_task(executor.store.row(job),demand=True)
                        poll = dict(host_id=HOST,slots=8,source_sha=source,cached_prefixes=[],
                                    host_fingerprints={job:cell['current_host_fingerprint']})
                        self.assertEqual(executor.rpc('/v1/poll',poll)['assignments'],[])
                        self.assertEqual(executor.store.row(job)['state'],'QUEUED')
                        self.assertFalse(list(run.rglob('LAUNCH.json')))
                        executor.distributed['host_admission']['hosts'][HOST]['accepting']=True
                        claimed = executor.rpc('/v1/poll',poll)['assignments']
                        self.assertEqual(len(claimed),1)
                        fixed = claimed[0]
                        self.assertEqual(fixed['request'],assignment['request'])
                        self.assertEqual(fixed['host_execution_profile'],want)
                        self.assertEqual(fixed['host_execution_profile_sha256'],Q.digest(want))
                        self.assertEqual(executor.rpc('/v1/poll',poll)['assignments'],claimed)
                        launch_bytes = next(run.glob('jobs/*/*/LAUNCH.json')).read_bytes()
                        profile = read(WSL_DOC / 'PROFILE.json')
                        profile.update(work_dir=str(directory / 'ARTIFICIAL-spool'),slots=1)
                        heads={str(Path(want['root']).resolve()):source}
                        heads.update({str(Path(view).resolve()):commit for commit,view in want['route_views'].items()})
                        filesystem={str(Path(want['root']).resolve()/name):raw for name,raw in original_bytes.items()}
                        original_read=Path.read_bytes;original_file=Path.is_file
                        def metadata_bytes(path):return filesystem.get(str(path),None) if str(path) in filesystem else original_read(path)
                        def metadata_file(path):return str(path) in filesystem or original_file(path)
                        def metadata_git(command,**kwargs):
                            self.assertEqual(command[:2],['git','show']);return input_blobs[command[2]]
                        def validate_profile(p):self.assertEqual(profile_core(p),want)
                        runtime_calls=[]
                        def metadata_runtime(p):
                            validate_profile(p);runtime_calls.append(profile_core(p))
                            fp=cell['current_host_fingerprint']
                            return dict(status='PASS',runtime=fp['runtime'],checks=fp.get('runtime_checks',[]))
                        def metadata_adapter(p):
                            validate_profile(p)
                            return dict(root=p['adapter_root'],**p['adapter_manifest']) if p.get('adapter_root') else None
                        commands=[]
                        class FakeProcess:
                            def __init__(self,command,**kwargs):
                                self.pid=99000000+len(commands);commands.append((list(command),kwargs))
                            def wait(self,*args,**kwargs):return 0
                            def poll(self):return 0
                        def fake_popen(command,**kwargs):
                            self.assertIn(Path(command[2]).name,['run_rpe_q3_worker.py','run_rpe_optimizer_q3_job.py'])
                            return FakeProcess(command,**kwargs)
                        evidence=cell['evidence'][0];readiness=read(evidence['readiness'])
                        self.assertEqual(hashlib.sha256(original_read(Path(evidence['readiness']))).hexdigest(),evidence['sha256'])
                        self.assertEqual(readiness['request_sha256'],fixed['request_sha256'])
                        self.assertEqual(readiness['command'][0],want['interpreter'])
                        self.assertIn('3.11.0rc1',cell['current_host_fingerprint']['runtime']['version'])
                        with patch.object(W,'head',side_effect=lambda path:heads[str(Path(path).resolve())]),\
                             patch.object(Path,'read_bytes',metadata_bytes),patch.object(Path,'is_file',metadata_file),\
                             patch.object(W.subprocess,'check_output',side_effect=metadata_git),\
                             patch.object(W.subprocess,'Popen',side_effect=fake_popen),\
                             patch.object(W,'validate_adapter_profile',side_effect=metadata_adapter),\
                             patch.object(W,'check_runtime_profile',side_effect=metadata_runtime):
                            worker=W.Worker(profile,executor.controller_url)
                            try:
                                worker._acquire(fixed)
                                attempt=worker.work/job/fixed['attempt_id']
                                stored=W.read(attempt/'WORKER_PROFILE.json')
                                self.assertEqual(profile_core(stored),want)
                                resolved=W.validate_assignment(stored,fixed)
                                self.assertEqual(resolved['route']['interpreter'],want['interpreter'])
                                self.assertEqual(resolved['route']['view'],want['route_views'][resolved['route']['view_commit']])
                                W.atomic(attempt/'RESOLVED_REQUEST.json',resolved)
                                worker._launch(attempt)
                                self.assertEqual(commands[-1][0][0],want['interpreter'])
                                self.assertEqual(W._run_adapter(stored,attempt,'execute',attempt),0)
                                command,kw=commands[-1]
                                self.assertEqual(command[0],want['interpreter'])
                                self.assertEqual(command[2],str(Path(want.get('adapter_root',want['root']))/'scripts/run_rpe_optimizer_q3_job.py'))
                                self.assertEqual(kw['cwd'],want.get('adapter_root',want['root']))
                                if want.get('adapter_manifest'):
                                    self.assertEqual(W.read(attempt/'ROUTE_VIEWS.json'),want['route_views'])
                                    self.assertEqual(W.read(attempt/'ADAPTER_SOURCE.json'),dict(root=want['adapter_root'],**want['adapter_manifest']))
                                    self.assertEqual(command[command.index('--route-views')+1],str(attempt/'ROUTE_VIEWS.json'))
                                if want.get('runtime_checks'):
                                    self.assertIn(want,runtime_calls)
                                    self.assertEqual(readiness['runtime_precheck']['checks'],cell['current_host_fingerprint']['runtime_checks'])
                                before=len(commands);worker._acquire(fixed)
                                self.assertEqual(len(commands),before)
                            finally:worker.close()
                            restarted=W.Worker(dict(profile,route_profiles=[]),executor.controller_url)
                            try:
                                restarted._acquire(fixed)
                                saved=W.read(attempt/'WORKER_PROFILE.json')
                                self.assertEqual(profile_core(saved),want)
                                self.assertEqual(W._adapter_command(saved,attempt,'execute',attempt)[0],want['interpreter'])
                                self.assertEqual(len(commands),before)
                            finally:restarted.close()
                        self.assertEqual(len(commands),2)
                        executor.rpc('/v1/state',dict(host_id=HOST,job_id=job,attempt_id=fixed['attempt_id'],state='RUNNING'))
                        executor.distributed['host_admission']['hosts'][HOST]['accepting']=False
                        self.assertEqual(executor.rpc('/v1/poll',poll)['assignments'],[])
                        self.assertEqual(executor.store.row(job)['attempt_id'],fixed['attempt_id'])
                        release(executor);executor=create()
                        self.assertEqual(executor.rpc('/v1/poll',poll)['assignments'],[])
                        self.assertEqual(executor.store.row(job)['host_id'],HOST)
                        self.assertEqual(executor.store.row(job)['attempt_id'],fixed['attempt_id'])
                        self.assertEqual(next(run.glob('jobs/*/*/LAUNCH.json')).read_bytes(),launch_bytes)
                        self.assertFalse(any(e['event'] in ('ACCEPT','ADOPT')for e in executor.store.events()))
                        self.assertFalse(list(run.rglob('OUTCOME.json')))
                        report_rows.append(dict(method=assignment['method'],job_id=job,request_sha256=fixed['request_sha256'],
                            profile_sha256=Q.digest(want),interpreter=want['interpreter'],route_views=want['route_views'],
                            adapter_manifest=want.get('adapter_manifest'),runtime_checks=want.get('runtime_checks',[]),
                            off_claims=0,joined_claims=1,max_slots=1,mock_supervisor_invocations=1,mock_adapter_invocations=1,
                            worker_restart_extra_launches=0,controller_restart_extra_launches=0,
                            final_accepting=False,outcomes=0,adoptions=0,actual_processes=0))
                    finally:release(executor)
        for name,raw in original_bytes.items():self.assertEqual((computation/name).read_bytes(),raw)
        result=dict(schema='RPE_WSL_PROFILE_ASSIGNMENTS_V1',status='PASS',
            scope='VERIFIED_REAL_REQUESTS_ARTIFICIAL_QUEUE_AND_MOCK_WORKER_INVOCATIONS',
            policy=str(policy_path),policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            rows=report_rows,verified_wsl_cells=len(report_rows),baseline_twelve_cells_reexecuted=0,
            simulated_claims=len(report_rows),actual_worker_or_adapter_processes=0,
            new_scientific_evaluations=0,production_queue_operations=0,
            metadata_stubs=['Remote HEAD/source files replay fixed SOURCE; input blobs read from local fixed checkout',
                'Runtime and adapter metadata replay verified evidence; all supervisor/adapter Popen calls mocked'],
            unverified_wsl_cells=[dict(method=row['method'],status=row['hosts'][HOST]['status'])
                for row in policy['jobs'].values()if row['hosts'][HOST]['status']!='PASS'])
        output=os.environ.get('RPE_WSL_PROFILE_ASSIGNMENT_ACCEPTANCE')
        if output:Q.atomic_json(Path(output),result)


if __name__=='__main__':unittest.main()
