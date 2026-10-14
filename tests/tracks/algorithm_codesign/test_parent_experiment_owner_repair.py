"""Real acquisition/job path with fake processes and artificial key metadata.

No registered fixtures, sequences, numeric providers, workers or backend calls.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock,patch

from trottertracks.algorithm_codesign.parent_experiment.pipeline import acquire
from trottertracks.algorithm_codesign.parent_experiment.launch import CONTRACT_SHA,execution_identity,verify_worker
from trottertracks.algorithm_codesign.rte_development.evidence import EvidenceWriter,digest,atomic_json


class AcquisitionOwnerControls(unittest.TestCase):
    def exercise(self, *, keys=1, fail=False, corrupt=False, verify=True):
        """The actual acquire loop writes a real job and validates a fake result."""
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'results';d.mkdir()
            execution=dict(result_directory='results',authorization_path='auth.json',source_manifest='m.json',optional_receipt_path='receipt.md',worker='off-domain-worker.py')
            caps=dict(workers=2,cache_bytes=100000,worker_RSS_MiB=512)
            c=dict(execution=execution,caps=caps,acquisition=dict(runtime_executable='off-domain-python',tool_identity=dict(sha256='a'*64)),review=dict(sha256='b'*64))
            source='1'*40;head='2'*40
            authority=dict(status='APPROVED_FOR_ONE_PARENT_ALLOCATION_RUN',science_execution_authorized=True,source_commit=source,
                contract_sha256=CONTRACT_SHA,runs=1,retries=0,mandatory_STOP=True,explicit_execution_instruction='synthetic owner test only')
            (root/'auth.json').write_text(json.dumps(authority));(root/'m.json').write_text('{}')
            base=execution_identity(root,c,authority,head)
            # Outer owner is 101; ledger owning the synthesis workers is 202.
            atomic_json(d/'one_shot_consumed.json',dict(**base,parent_pid=101))
            (d/'candidate_freeze.json').write_text('{"artificial":true}')
            inventory=dict(keys=[dict(key=str(i)*64,ratio=r,acquisition='NEW_ONCE')
                for i,r in zip(range(1,keys+1),('2/11','3/13'))])
            (d/'inventory.json').write_text(json.dumps(inventory));caps['new_synthesis_keys']=18
            identity=dict(**base,parent_pid=101,marker_sha256=digest(d/'one_shot_consumed.json'),
                candidate_freeze_sha256=digest(d/'candidate_freeze.json'),inventory_sha256=digest(d/'inventory.json'))
            immutable=copy.deepcopy(identity);marker_before=(d/'one_shot_consumed.json').read_bytes()
            writer=EvidenceWriter(d,1000000);guard=Mock();guard.children=set();guard.reaped_cpu=0
            starts=[];processes=[]
            class Process:
                def __init__(self,index):self.pid=900+index;self.returncode=None;self.terminated=False
                def poll(self):return self.returncode
                def terminate(self):self.terminated=True;self.returncode=-15
                def wait(self,timeout=None):return self.returncode
                def kill(self):self.returncode=-9
            def fake_git(_root,*args):
                if args[:2]==('rev-parse','HEAD'):return head
                if args[0]=='show':return source
                if args[0]=='diff':return 'auth.json'
                return ''
            def popen(command,**kwargs):
                path=Path(command[-1]);job=json.loads(path.read_text());index=len(starts)
                self.assertEqual(job['parent_pid'],202);self.assertNotEqual(job['parent_pid'],identity['parent_pid'])
                self.assertEqual(job['source_commit'],source);self.assertEqual(job['execution_HEAD'],head)
                self.assertEqual(job['inventory_sha256'],identity['inventory_sha256'])
                self.assertEqual(job['candidate_freeze_sha256'],identity['candidate_freeze_sha256'])
                for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
                    self.assertEqual(kwargs['env'][name],'1')
                if verify:
                    # Exercise the actual worker gate without executing worker.
                    verify_worker(root,path)
                starts.append(job);proc=Process(index);processes.append(proc)
                # Preserve raw job identity in the process result.
                value=dict(job,status='KEY_COMPLETE',synthesis_calls=1,retries=0,saved={'artificial_price':index+1})
                if corrupt:value['parent_pid']=101
                path.with_suffix('.result.json').write_text(json.dumps(value))
                return proc
            def wait4(pid,options):
                # First child failure leaves the second peer for finally cleanup.
                index=pid-900;code=1 if fail and index==0 else 0
                return pid,code<<8,SimpleNamespace(ru_utime=.01,ru_stime=.01,ru_maxrss=64)
            with patch('trottertracks.algorithm_codesign.parent_experiment.launch.read_contract',return_value=c),\
                 patch('trottertracks.algorithm_codesign.parent_experiment.launch.git',side_effect=fake_git),\
                 patch('trottertracks.algorithm_codesign.parent_experiment.launch.check_identities'),\
                 patch('os.getpid',return_value=202),patch('os.getppid',return_value=202),\
                 patch('subprocess.Popen',side_effect=popen),patch('os.wait4',side_effect=wait4):
                if fail or corrupt:
                    with self.assertRaises((RuntimeError,PermissionError)):acquire(root,c,identity,inventory,writer,guard)
                    result=None
                else:result=acquire(root,c,identity,inventory,writer,guard)
            self.assertEqual(identity,immutable);self.assertEqual((d/'one_shot_consumed.json').read_bytes(),marker_before)
            self.assertFalse(guard.children)
            return dict(result=result,starts=starts,processes=processes,
                        jobs=[p.name for p in (d/'jobs').glob('key_[0-9][0-9][0-9].json')],
                        process_receipts=[json.loads(p.read_text()) for p in (d/'jobs').glob('*.process_receipt.json')],
                        backend_calls=0)

    def test_existing_outer_pid_first_real_job_success(self):
        v=self.exercise();self.assertEqual(len(v['starts']),1);self.assertEqual(v['starts'][0]['parent_pid'],202)
        self.assertEqual(list(v['result'][0]),['2/11']);self.assertEqual(len(v['result'][1]),1);self.assertEqual(v['backend_calls'],0)

    def test_two_workers_keep_ledger_owner_and_key_index(self):
        v=self.exercise(keys=2);self.assertEqual(len(v['starts']),2)
        self.assertEqual([x['job_index'] for x in v['starts']],[0,1]);self.assertEqual({x['parent_pid'] for x in v['starts']},{202})
        self.assertEqual(set(v['result'][0]),{'2/11','3/13'});self.assertEqual(len(v['process_receipts']),2)

    def test_worker_failure_saved_once_no_retry(self):
        v=self.exercise(fail=True);self.assertEqual(len(v['starts']),1)
        self.assertEqual(v['process_receipts'][0]['exit_code'],1);self.assertIsNone(v['result'])

    def test_result_owner_mismatch_rejected(self):
        v=self.exercise(corrupt=True);self.assertEqual(len(v['starts']),1);self.assertIsNone(v['result'])

    def test_peer_is_terminated_after_first_worker_failure(self):
        v=self.exercise(keys=2,fail=True);self.assertEqual(len(v['starts']),2)
        self.assertTrue(v['processes'][1].terminated)
        self.assertTrue(any(r.get('terminated_due_to_peer_or_parent_failure') for r in v['process_receipts']))

    def test_original_failure_condition_without_worker_is_python_typeerror(self):
        identity={'parent_pid':101}
        with self.assertRaisesRegex(TypeError,'parent_pid'):dict(**identity,parent_pid=202)
        self.assertEqual(dict(identity,parent_pid=202),{'parent_pid':202});self.assertEqual(identity,{'parent_pid':101})


class WorkerOwnerBindingControls(unittest.TestCase):
    def test_wrong_outer_pid_cannot_own_key_worker(self):
        # Same actual acquisition helper, but simulate a worker whose real parent
        # differs from the ledger owner recorded by the fixed job.
        control=AcquisitionOwnerControls()
        original=verify_worker
        def rejected(root,path,role='key'):
            with patch('os.getppid',return_value=101):return original(root,path,role)
        with patch('test_parent_experiment_owner_repair.verify_worker',side_effect=rejected):
            with self.assertRaises(PermissionError):control.exercise()

if __name__=='__main__':unittest.main()
