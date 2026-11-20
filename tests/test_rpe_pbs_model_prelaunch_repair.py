"""Synthetic PBS transport rejection: preserve UNKNOWN and prove no stage start."""
import hashlib
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

import test_rpe_metadata_repair_continuation as base
from trottertracks.rpe_release_registry import fingerprint
from trottertracks.rpe_search_continuation import verify, verify_pbs_prelaunch_rejections


class PBSModelPrelaunchRepairTests(unittest.TestCase):
    save = base.MetadataRepairTests.save
    bind_error = base.MetadataRepairTests.bind_error

    def setUp(self):
        base.MetadataRepairTests.setUp(self)
        self.save('STOP_REQUESTED.json', {'reason':'USER_REQUESTED_STOP'})
        self.save('KAPPA_STATE.json', {'stop':{'reason':'USER_REQUESTED_STOP'}})
        self.save('EXIT.json', {'invocation_id':'owned','returncode':0})
        (self.original/'completion').mkdir()
        self.save('completion/STATE.json', {'records':{}})
        (self.original/'distributed/relay').mkdir()
        self.save('distributed/relay/STATUS.json', {'stopped':True})
        (self.original/'distributed/pbs').mkdir()
        self.save('distributed/pbs/STATE.json', {'allocations':{'slot':{
            'status':'F','observed':{'job_id':'17.pbs'},'spec':{'profile':{'host_id':'pbs-0','work_dir':'/remote/owned'}}}}})
        self.request={'original_contract':{'stage':'PF_MODEL_FREEZE'},
                      'preparation':{'stage':'audit','provider':'PF_MODEL_FREEZE'}}
        self.detail={'reason':'LAUNCH_STATE_UNCERTAIN_NO_ADAPTER_EXIT_RECEIPT','automatic_retry':False}
        digest=fingerprint(self.request)
        unknown=('prep-rejected','PREPARATION:audit','UNKNOWN','pbs-0','blocked',digest)
        with sqlite3.connect(self.original/'distributed/QUEUE.sqlite3')as db:
            for name in ('request','detail','completion_sha256'):
                db.execute('ALTER TABLE jobs ADD COLUMN '+name)
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)',(*unknown,json.dumps(self.request),json.dumps(self.detail),None))
            self.rows=db.execute('SELECT job_id,method,state,host_id,attempt_id,request_sha256 FROM jobs ORDER BY job_id').fetchall()
        self.job=self.original/'jobs/PREPARATION:audit/prep-rejected';self.job.mkdir(parents=True)
        self.save('jobs/PREPARATION:audit/prep-rejected/REQUEST.json',self.request)
        self.save('jobs/PREPARATION:audit/prep-rejected/LAUNCH.json',{'host_id':'pbs-0','attempt_id':'blocked'})
        self.evidence=dict(kind='PF_MODEL_FREEZE_WRAPPER_REJECTION',prior_run=str(self.original),job_id='17.pbs',job_state='F',
            attempt='/remote/owned/prep-rejected/blocked',
            assignment=dict(job_id='prep-rejected',attempt_id='blocked',request=self.request,request_sha256=digest),
            worker_state=dict(state='UNKNOWN',detail=self.detail),
            files=['ASSIGNMENT.json','SUPERVISOR_LAUNCH.json','WORKER_SUPERVISOR.log','WORKER_STATE.json'],
            supervisor_log='Traceback (synthetic)\nValueError: Unsupported fixed PF entry: run_rpe_q3_worker.py\n',
            supervisor_return=dict(event='SUPERVISOR_RETURN',host_id='pbs-0',job_id='prep-rejected',attempt_id='blocked',returncode=1,signal=None))
        self.round['continuation'].update(kind='OPERATIVE_PBS_PRELAUNCH_REPAIR',parent_queue_sha256=fingerprint(self.rows),
            parent_stop_sha256=hashlib.sha256((self.original/'STOP_REQUESTED.json').read_bytes()).hexdigest(),
            parent_metadata_state_sha256=hashlib.sha256((self.original/'completion/STATE.json').read_bytes()).hexdigest(),
            metadata_hold_keys=[],pbs_prelaunch_rejections=[self.evidence])

    def test_explicit_repair_preserves_unknown_and_has_no_forged_exit_receipt(self):
        queue=self.original/'distributed/QUEUE.sqlite3';before=queue.read_bytes()
        result=verify(self.successor,self.problem,self.round)
        self.assertEqual(result['consumed'],{'PREPARATION:signal':1,'PR_PAULI':1})
        self.assertEqual(queue.read_bytes(),before)
        self.assertEqual({p.name for p in self.job.iterdir()},{'REQUEST.json','LAUNCH.json'})
        self.successor.mkdir()
        (self.successor/'ROUND.requested.json').write_text(json.dumps(self.round))
        self.assertEqual(verify_pbs_prelaunch_rejections(self.successor,[str(self.original)]),{self.job.resolve()})

    def test_started_stage_or_live_owner_is_never_replayed(self):
        for marker in ('SCIENTIFIC_STARTING.json','SUPERVISOR_STARTED.json','PREPARATION_STARTED.json','PROCESS_RETURN.json','OUTCOME.json'):
            with self.subTest(marker=marker):
                self.evidence['files'].append(marker)
                with self.assertRaises(ValueError):verify(self.successor,self.problem,self.round)
                self.evidence['files'].pop()
        with patch('trottertracks.rpe_search_continuation._has_local_owner',return_value=True):
            with self.assertRaises(ValueError):verify(self.successor,self.problem,self.round)

    def test_terminal_scheduler_and_actual_transport_return_are_required(self):
        self.evidence['job_state']='R'
        with self.assertRaises(ValueError):verify(self.successor,self.problem,self.round)
        self.evidence['job_state']='F';self.evidence['supervisor_return']['returncode']=None
        with self.assertRaises(ValueError):verify(self.successor,self.problem,self.round)
        self.evidence['supervisor_return']['returncode']=1
        self.evidence['supervisor_return']['attempt_id']='another-attempt'
        with self.assertRaises(ValueError):verify(self.successor,self.problem,self.round)

    def test_other_continuation_and_other_unknown_job_still_hold(self):
        self.round['continuation']['kind']='OPERATIVE_DEPLOYMENT_REPAIR'
        with self.assertRaises(ValueError):verify(self.successor,self.problem,self.round)
        self.round['continuation']['kind']='OPERATIVE_PBS_PRELAUNCH_REPAIR'
        with sqlite3.connect(self.original/'distributed/QUEUE.sqlite3')as db:
            db.execute("UPDATE jobs SET state='UNKNOWN' WHERE job_id='b'")
            rows=db.execute('SELECT job_id,method,state,host_id,attempt_id,request_sha256 FROM jobs ORDER BY job_id').fetchall()
        self.round['continuation']['parent_queue_sha256']=fingerprint(rows)
        with self.assertRaisesRegex(ValueError,'unresolved or unknown'):verify(self.successor,self.problem,self.round)
