"""Synthetic operational checks of the shared latency/storage fixes."""
import base64
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trottertracks import rpe_pbs as P, rpe_q3_coordinator as Q, rpe_worker_storage as S
from trottertracks.rpe_preparation_artifacts import ArtifactCache
from trottertracks.rpe_pbs_stream import SpoolReader, canonical_digest
from compact_rpe_git_objects import compact
from publish_h4_search_run import prepare

class EfficiencyTests(unittest.TestCase):
    def test_fast_storage_report_requests_early_sweep_and_proactive_uses_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);policy=dict(limit_bytes=1000,roots=[tmp],sample_path=str(root/'usage.json'))
            with patch.object(S,'_used_bytes',return_value=810), patch.object(S,'_reclaim',return_value=(700,1)) as reclaim:
                fast=S.report(policy,spools=(tmp,))
                self.assertEqual((fast['reclaim_start_bytes'],fast['reclaim_target_bytes']), (800,700))
                self.assertTrue(fast['accepting_new_jobs']);self.assertTrue(fast['reclaim_pending']);reclaim.assert_not_called()
                watched=S.report(policy,spools=(tmp,),proactive=True)
                self.assertEqual(watched['used_bytes'],700);self.assertFalse(watched['reclaim_pending'])
                reclaim.assert_called_once()

    def test_large_queue_diagnostic_keeps_counts_and_only_changed_jobs(self):
        from trottertracks.rpe_optimizer_concurrent import DistributedExecutor
        ex=DistributedExecutor.__new__(DistributedExecutor)
        ex.distributed={};ex.hosts={'local':dict(capacity=1,slots=1)};ex.capacity=10
        tasks={Path(f'job-{i:05}'):SimpleNamespace(job=Path(f'job-{i:05}'),status='QUEUED',rule={'kind':'q3'}) for i in range(15000)}
        ex.state=SimpleNamespace(tasks=tasks,active=0)
        rows={task.job.name:dict(job_id=task.job.name,eligibility=None) for task in tasks.values()}
        ex.store=SimpleNamespace(rows_for=lambda jobs:{},_connection=lambda:None)
        ex._owned_attempts=lambda:[]
        ex._host_admission_policy=lambda: (None,None)
        logs=[];ex.log=lambda **kw:logs.append(kw)
        ex._report_queue_snapshot(observed_queue=list(tasks.values()),observed_rows=rows,observed_owned=[])
        self.assertEqual(len(logs[-1]['reasons']),15000)
        tasks[Path('job-00000')].status='RUNNING'
        ex._report_queue_snapshot(observed_queue=list(tasks.values()),observed_rows=rows,observed_owned=[])
        self.assertTrue(logs[-1]['reason_delta']);self.assertEqual(logs[-1]['reasons'],{})
        self.assertEqual(logs[-1]['reason_removed'],['job-00000'])
        self.assertEqual(sum(logs[-1]['reason_counts'].values()),14999)

    def test_pbs_finished_specs_are_written_once_and_restored_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager=P.PbsManager.__new__(P.PbsManager);manager.path=Path(tmp)/'STATE.json'
            spec=dict(token='old',run_id='test',profile=dict(host_id='pbs',work_dir='/old',large='x'*200000),reason={})
            active=dict(token='new',profile=dict(host_id='pbs',large='y'*100))
            manager.state=dict(allocations={'old':dict(status='F',harvested=True,spec=spec),
                                           'new':dict(status='R',spec=active)},requests={})
            manager.save();saved=json.loads(manager.path.read_text());self.assertLess(manager.path.stat().st_size,2000)
            archived=manager.path.parent/'allocation-specs'/(saved['allocations']['old']['archived_spec']+'.json');before=archived.stat().st_mtime_ns
            manager.state['heartbeat']=1;manager.save();self.assertEqual(archived.stat().st_mtime_ns,before)
            self.assertEqual(P.load_pbs_manager_state(manager.path),manager.state)
            self.assertEqual(saved['allocations']['old']['spec']['profile']['work_dir'],'/old')

    def test_streamed_wire_identity_and_exact_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=b'abc\x00'*400000+b'12'
            files={'a/日本.json':base64.b64encode(raw).decode(),'empty':'','one':'YQ=='}
            request=dict(request_id='test',endpoint='/v1/complete',payload=dict(files=files,host_id='pbs',job_id='j',attempt_id='a',request_sha256='r'),clock=dict(monotonic=1.5,other=[True,None,'x\\n']))
            wire=json.dumps(request).encode();parsed=SpoolReader((wire[i:i+131071] for i in range(0,len(wire),131071)),root).read()
            staged={name:blob.path for name,blob in parsed['payload']['files'].items()}
            self.assertEqual(canonical_digest(parsed),Q.digest(request));self.assertEqual(Q.staged_completion_digest(staged),Q.digest(files))
            target=root/'target';Q.install_staged_files(target,staged);Q.install_staged_files(target,staged)
            self.assertEqual((target/'a/日本.json').read_bytes(),raw)
            self.assertEqual((target/'empty').stat().st_size,0)

    def test_streamed_artifact_rpc_retains_original_response_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manager=P.PbsManager.__new__(P.PbsManager);manager.directory=root
            cache=ArtifactCache(root/'cache');manager.executor=SimpleNamespace(artifact_cache=cache,_receive_completion=lambda *a,**k:None)
            manager.state=dict(allocations={'token':dict(spec=dict(profile=dict(host_id='pbs')))})
            raw=b'z'*1100000;sha=hashlib.sha256(raw).hexdigest()
            req=dict(request_id='artifact-put-1',endpoint='/v1/artifacts/put',payload=dict(host_id='pbs',blobs={sha:base64.b64encode(raw).decode()}))
            wire=json.dumps(req).encode();manager.bulk_gateway=None
            manager._iter_bulk=lambda *args,**kw:(wire[i:i+65537] for i in range(0,len(wire),65537))
            desc=dict(request_id=req['request_id'],spool_bytes=len(wire))
            reply=manager._dispatch_streamed_spool('token',desc);self.assertEqual(reply['result']['bytes'],len(raw))
            self.assertEqual(manager._dispatch_streamed_spool('token',desc),reply)
            saved=json.loads((root/'responses/token/artifact-put-1.json').read_text());self.assertEqual(saved['request_sha256'],Q.digest(req));self.assertEqual(cache.path(sha).read_bytes(),raw)

    def test_native_git_compaction_keeps_every_selected_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=root/'repo';repo.mkdir()
            def git(*args,**kwargs):return subprocess.check_output(['git','-C',str(repo),*args],**kwargs)
            git('init','-q');git('remote','add','origin','git@github.com:HIROMU1015/synthetic.git')
            common=random.Random(7).randbytes(256*1024);blobs={}
            for i in range(4):
                raw=common+str(i).encode();oid=git('hash-object','-w','--stdin',input=raw).decode().strip();blobs[oid]=raw
            result=compact(repo,root/'maintenance',min_age=0,minimum_object_bytes=1,force=True)
            self.assertEqual(result['status'],'PACKED_VERIFIED');self.assertGreater(result['reclaimed_bytes'],0)
            for oid, raw in blobs.items():self.assertEqual(git('cat-file','blob',oid),raw)

    def test_live_estimate_not_a_final_result_in_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'docs/research/run';run.mkdir(parents=True)
            (run/'SCOPE_METADATA.json').write_text('{}')
            (run/'LIVE_SHORTLIST.json').write_text(json.dumps(dict(schema='RPE_LIVE_SHORTLIST_V1',points=[dict(method='PR_DF',M=12,estimated_total_RZ=5)],estimated_order=[])))
            (run/'LIVE_SEARCH_PROGRESS.json').write_text(json.dumps(dict(schema='H4_SEARCH_PROGRESS_V1',status='RECORDED',point_checkpoint_seq=3,families=[])))
            status,manifest=prepare(root,run,run/'SEARCH_PUBLICATION.json',run_id='run')
            self.assertEqual(status,'UPDATED');self.assertEqual(manifest['live_shortlist']['status'],'LIVE_EXPLORATORY_ESTIMATES')
            self.assertEqual(manifest['search_progress']['point_checkpoint_seq'],3)
            self.assertNotIn('ROUND_RESULT.json',manifest['files'])
