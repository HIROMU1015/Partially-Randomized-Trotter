import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from trottertracks.rpe_preparation_repair import retry_records, verify_attempts
from trottertracks.rpe_release_registry import fingerprint


class PreparationRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)/'parent';self.parent.mkdir()
        self.put('STOP_REQUESTED.json', {'reason':'USER_REQUESTED_STOP'})
        self.put('KAPPA_STATE.json', {'stop':{'reason':'USER_REQUESTED_STOP'}})
        self.put('EXIT.json', {'invocation_id':'fixed','returncode':0})
        self.put('LAUNCH.json', {'invocation_id':'fixed'})
        self.put('distributed/relay/STATUS.json', {'stopped':True})
        self.request={'request_id':'task','preparation':{'provider':'DF_DESCRIPTOR','dependencies':[]}}
        self.sha=fingerprint(self.request)
        self.job='jobs/PREPARATION:signal/task'
        self.put(self.job+'/REQUEST.json', self.request)
        self.put(self.job+'/OUTCOME.json', {'status':'PREPARATION_FAILED','automatic_retry':False,'request_sha256':self.sha})
        self.put(self.job+'/PREPARATION_RECEIPT.json', {'returncode':1,'signal':None,'automatic_retry':False})
        self.put(self.job+'/LAUNCH.json', {'host_id':'local','attempt_id':'one'})
        self.db=sqlite3.connect(self.parent/'distributed/QUEUE.sqlite3')
        self.addCleanup(self.db.close)
        self.db.execute('CREATE TABLE jobs(job_id,method,state,host_id,attempt_id,request_sha256,request)')
        self.db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?)', ('task','PREPARATION:signal','ACCEPTED','local','one',self.sha,json.dumps(self.request)))
        self.db.commit()
        self.original={'method':'PR_DF','status':'HELD_WITH_REASON','scientific_failure':True,'reason':'failed'}
        self.put('completion/STATE.json', {'records':{'selected':self.original,'other':dict(self.original)}})
        self.put('completion/candidates/selected/DF_DESCRIPTOR.DISTRIBUTED.json',
                 {'job_id':'task','request':{'preparation':{'provider':'DF_DESCRIPTOR'}}})
        self.repair={'state_file':str(self.parent/'completion/STATE.json'),'record_keys':['selected'],'job_ids':['task']}

    def put(self, name, value):
        p=self.parent/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))

    def test_selected_failure_archived_other_failure_preserved_and_originals_unchanged(self):
        before={p:p.read_bytes() for p in self.parent.rglob('*.json')}
        records, archive=retry_records({'selected':self.original,'other':self.original},[self.repair])
        self.assertNotIn('selected',records);self.assertEqual(records['other'],self.original)
        self.assertEqual(archive['selected']['original_attempts']['task']['attempt_id'],'one')
        self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_unknown_attempt_cannot_be_retried(self):
        self.db.execute("UPDATE jobs SET state='UNKNOWN'");self.db.commit()
        with self.assertRaisesRegex(ValueError,'unresolved'):
            verify_attempts(self.parent,['task'])

    def test_success_and_unended_parent_cannot_be_replayed(self):
        self.put(self.job+'/PREPARATION_RECEIPT.json', {'returncode':0,'automatic_retry':False})
        with self.assertRaisesRegex(ValueError,'failed receipt'):
            verify_attempts(self.parent,['task'])
        self.put('distributed/relay/STATUS.json', {'stopped':False})
        with self.assertRaisesRegex(ValueError,'ended normal-stop'):
            verify_attempts(self.parent,['task'])

    def test_request_or_candidate_binding_must_match(self):
        self.put('completion/candidates/selected/DF_DESCRIPTOR.DISTRIBUTED.json', {'job_id':'task','request':{'different':True}})
        with self.assertRaisesRegex(ValueError,'binding differs'):
            retry_records({'selected':self.original},[self.repair])

    def test_empty_retry_selection_does_not_lift_a_hold(self):
        with self.assertRaisesRegex(ValueError,'selected'):
            verify_attempts(self.parent,[])
