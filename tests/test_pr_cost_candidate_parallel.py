"""Synthetic scheduling/native-receipt fixtures, no scientific imports or compile."""
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import pr_cost_batch as b
import pr_cost_candidate_parallel as p
c=b.old


def task(cid,replica):
    return dict(id=cid+'_'+str(replica),cell=dict(id=cid,q=2,R=4),replica=replica,seed=100+replica+ord(cid[0]))


class Fixtures(unittest.TestCase):
    def setUp(self):
        base=Path('/home/AbeHiromu/projects/h8-pr-candidate-parallel-preparation/20261018/tmp')
        base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base,prefix='synthetic-');self.root=Path(self.temp.name)
        self.binding={'compiler':'c'*64,'system':'SYNTHETIC'}
        self.origin=dict(origin_output=str(self.root/'old'),origin_SOURCE='a'*40,
                         origin_parent_identity=[987654,1,999999,os.getuid()])
        self.tasks=[task('A',0),task('A',1)]
    def tearDown(self):self.temp.cleanup()
    def worker(self,item,*,code=0):
        directory=self.root/'old'/('first_control' if item['replica']==0 else 'remaining')/item['id']
        directory.mkdir(parents=True);identity=[987650+item['replica'],10+item['replica'],987654,os.getuid()]
        c.save(directory/'started.json',dict(pid=identity[0],identity=identity,task=item['id']))
        if code==0:
            events=[[{'synthetic':item['seed']}]];event=c.digest(events)
            c.save(directory/'trajectory.json',dict(cell=item['cell'],seed=item['seed'],events=events,event_digest=event,
                cohort='EXPLORATION_N8',conditional=False,quantum_weights_changed=False))
            axes={}
            for axis in ('cosine','sine'):
                axes[axis]=dict(RZ=10,CX=2,depth=4,RZ_depth=3,CX_depth=1,size=20,
                    seed=item['seed'],event_digest=event,compiler_hash=self.binding['compiler'],fingerprint='f'*64)
                c.save(directory/('measured_'+axis+'.json'),axes[axis])
            calls=dict(trajectory=1,occurrence=2,RTE_event=4,native_evolution=1,wrapper_compile=2)
            if item['replica']==0:
                calls.update(control_probe=4,independent_event_oracle=1)
                c.save(directory/'control.json',dict(status='PASS',max_error=0.,tolerance=1e-8,
                    control_probes=4,independent_event_oracles=1,full_Fock_scalar_once=True,
                    reference_rotated=False,ordinary_ancilla_branch=True))
            for kind,total in calls.items():
                c.save(directory/(kind+'_'+str(total)+'_reserved.json'),dict(kind=kind,count=total,total=total,nonrefundable=True))
            c.save(directory/'task_result.json',dict(cell_id=item['cell']['id'],replica=item['replica'],
                seed=item['seed'],binding=self.binding,cohort='EXPLORATION_N8',synthetic=True,
                normalization_unchanged=True,event_digest=event,axes=axes,calls=calls))
        else:c.save(directory/'trajectory_1_reserved.json',dict(kind='trajectory',count=1,total=1,nonrefundable=True))
        return dict(task_id=item['id'],directory=str(directory),pid=identity[0],identity=identity,
            reaped=True,exit_code=code,result_sha256=c.sha(directory/'task_result.json') if code==0 else None)
    def receipt(self,item=None,*,code=0,status='PASS',error=None):
        item=item or self.tasks[0];worker=self.worker(item,code=code)
        path=Path(worker['directory']).parent/'parallel_receipt.json'
        c.save(path,dict(status=status,all_reaped=True,cleanup_errors=[],first_error=error,workers=[worker]))
        return path
    def manifest(self,receipt,**kw):
        return p.build_manifest(self.origin,self.tasks,self.binding,[receipt],synthetic=True,
                                identity_fn=lambda pid:None,**kw)
    def test_local_control_releases_only_its_candidate(self):
        tasks=[task(cid,r) for cid in ('A','B') for r in (0,1)]
        q=p.CandidateQueue(tasks)
        self.assertFalse(q.ready(tasks[1]));q.finish(tasks[0])
        self.assertTrue(q.ready(tasks[1]));self.assertFalse(q.ready(tasks[3]))
    def test_reused_control_unlocks_seven_without_redraw(self):
        tasks=[task('A',i) for i in range(8)];q=p.CandidateQueue(tasks,['A_0'])
        self.assertEqual(len(q.pending()),7);self.assertTrue(all(q.ready(t) for t in q.pending()))
        self.assertEqual([t['seed'] for t in q.pending()],[t['seed'] for t in tasks[1:]])
    def test_reused_dependent_without_control_rejected(self):
        with self.assertRaisesRegex(ValueError,'REUSED_DEPENDENTS'):p.CandidateQueue(self.tasks,['A_1'])
    def test_duplicate_completion_rejected(self):
        q=p.CandidateQueue(self.tasks);q.finish(self.tasks[0])
        with self.assertRaisesRegex(ValueError,'ONE_COMPLETION'):q.finish(self.tasks[0])
    def test_longer_representatives_start_first_without_changing_seeds(self):
        a=task('A',0);z=task('Z',0);z['cell'].update(R=64,q=16)
        frozen=deepcopy([a,z]);queue=p.CandidateQueue([a,z])
        self.assertEqual(queue.pending()[0]['id'],'Z_0');self.assertEqual([a,z],frozen)
    def test_live_origin_cannot_be_accepted(self):
        receipt=self.receipt()
        with self.assertRaisesRegex(ValueError,'ORIGIN_STILL_RUNNING'):
            p.build_manifest(self.origin,self.tasks,self.binding,[receipt],synthetic=True,
                identity_fn=lambda pid:tuple(self.origin['origin_parent_identity']))
    def test_exit_zero_reaped_receipt_reuse(self):
        manifest=self.manifest(self.receipt());self.assertEqual(set(manifest['accepted']),{'A_0'})
        self.assertEqual(manifest['remaining_task_ids'],['A_1'])
        p.verify_manifest(manifest,self.tasks,self.binding,self.origin,synthetic=True,identity_fn=lambda pid:None)
    def test_result_file_without_native_parent_receipt_rejected(self):
        self.worker(self.tasks[0])
        with self.assertRaises(ValueError):self.manifest(self.root/'missing.json')
    def test_reap_false_rejected(self):
        path=self.receipt();value=c.read(path);value['all_reaped']=False;path.write_text(__import__('json').dumps(value))
        with self.assertRaisesRegex(ValueError,'NATIVE_CLEANUP'):self.manifest(path)
    def test_nonzero_exit_never_reused_and_charge_preserved(self):
        path=self.receipt(code=2,status='STOP',error='OWN_RUN_SIGNAL_15')
        auth=self.root/'handoff.json';c.save(auth,dict(approved=True))
        h=dict(handoff_authorized=True,origin_SOURCE='a'*40,receipt=p.file_row(auth))
        manifest=self.manifest(path,handoff=h)
        self.assertFalse(manifest['accepted']);self.assertFalse(manifest['old_attempts'][0]['failed_or_unfinished_charge_refunded'])
        self.assertTrue(manifest['old_attempts'][0]['reservations'])
    def test_scientific_stop_cannot_be_accepted_as_handoff(self):
        path=self.receipt(status='STOP',error='NATIVE_SCALAR_CONTROL_EVENT_ORACLE')
        auth=self.root/'handoff.json';c.save(auth,{})
        with self.assertRaisesRegex(ValueError,'ONLY_EXPLICIT_HANDOFF'):
            self.manifest(path,handoff=dict(handoff_authorized=True,origin_SOURCE='a'*40,receipt=p.file_row(auth)))
    def test_mutated_result_refused(self):
        path=self.receipt();manifest=self.manifest(path)
        value=manifest['accepted']['A_0'];r=Path(value['directory'])/'measured_sine.json';r.write_text('{}')
        with self.assertRaisesRegex(ValueError,'INPUT_BYTES_CHANGED'):
            p.verify_manifest(manifest,self.tasks,self.binding,self.origin,synthetic=True,identity_fn=lambda pid:None)
    def test_native_result_checksum_mismatch_rejected(self):
        path=self.receipt();value=c.read(path);value['workers'][0]['result_sha256']='0'*64
        path.write_text(__import__('json').dumps(value))
        with self.assertRaisesRegex(ValueError,'NATIVE_RESULT_SHA'):self.manifest(path)
    def test_missing_native_result_hash_is_redone_without_refund(self):
        path=self.receipt();value=c.read(path);value['workers'][0]['result_sha256']=None
        path.write_text(__import__('json').dumps(value));manifest=self.manifest(path)
        self.assertFalse(manifest['accepted']);self.assertEqual(len(manifest['remaining_task_ids']),2)
        self.assertTrue(manifest['old_attempts'][0]['reservations'])
        self.assertFalse(manifest['old_attempts'][0]['failed_or_unfinished_charge_refunded'])
    def test_manifest_cannot_omit_a_required_file(self):
        manifest=self.manifest(self.receipt());manifest['files']=manifest['files'][:-1]
        value=dict(manifest);value.pop('manifest_digest');manifest['manifest_digest']=c.digest(value)
        with self.assertRaisesRegex(ValueError,'RECHECKED_NATIVE_ACCEPTANCE:files'):
            p.verify_manifest(manifest,self.tasks,self.binding,self.origin,synthetic=True,identity_fn=lambda pid:None)
    def test_invalid_control_never_unlocks_candidate(self):
        path=self.receipt();directory=path.parent/'A_0';value=c.read(directory/'control.json')
        value['max_error']=value['tolerance'];(directory/'control.json').write_text(__import__('json').dumps(value))
        with self.assertRaisesRegex(ValueError,'NATIVE_CONTROL_REQUIRED'):self.manifest(path)
    def test_seed_changed_refused(self):
        path=self.receipt();changed=deepcopy(self.tasks);changed[0]['seed']+=1
        with self.assertRaisesRegex(ValueError,'FROZEN_TASK_RESULT'):
            p.build_manifest(self.origin,changed,self.binding,[path],synthetic=True,identity_fn=lambda pid:None)
    def test_synthetic_receipt_never_accepted_as_real(self):
        path=self.receipt()
        with self.assertRaisesRegex(ValueError,'UNCHANGED_TRAJECTORY_COHORT'):
            p.build_manifest(self.origin,self.tasks,self.binding,[path],identity_fn=lambda pid:None)
    def test_invalid_handoff_grant_refused(self):
        spec=dict(scope='H8_A_COST',technical_sealed=True,approved=False,runtime_authorization=False,
            retry=False,next_stage_authorized=False,A_scheduling=p.POLICY)
        spec['technical_digest']=b.metadata_binding(spec)
        with self.assertRaisesRegex(ValueError,'HANDOFF_APPROVAL'):b.authorize(spec,{})
    def test_live_scheduler_runs_other_replica_before_slow_control(self):
        tasks=[task(cid,r) for cid in ('A','B') for r in (0,1)];queue=p.CandidateQueue(tasks);release=self.root/'released'
        def callback(t,directory):
            if t['id']=='B_0':
                until=time.monotonic()+3
                while not release.exists() and time.monotonic()<until:time.sleep(.01)
                if not release.exists():raise RuntimeError('GLOBAL_BARRIER_STILL_PRESENT')
            if t['id']=='A_1':release.write_text('A replica ran before B control completed')
            return dict(task=t['id'])
        def complete(t,h):
            self.assertTrue(h['reaped']);self.assertEqual(h['exit_code'],0);queue.finish(t)
        receipt=p.pool.map_owned(queue.pending(),callback,cpus=sorted(os.sched_getaffinity(0))[:2],
            limits={'max_workers':2},directory=self.root/'scheduler',deadline=math.inf,
            ready=queue.ready,on_complete=complete)
        self.assertTrue(release.exists());self.assertTrue(receipt['all_reaped'])
        self.assertEqual(len(queue.completed),4)
    def test_failed_control_reaps_owned_children_without_dependents(self):
        tasks=[task(cid,r) for cid in ('A','B') for r in (0,1)];queue=p.CandidateQueue(tasks)
        def callback(t,directory):
            if t['id']=='A_0':raise RuntimeError('ARTIFICIAL_CONTROL_FAIL')
            time.sleep(3);return {}
        with self.assertRaisesRegex(RuntimeError,'ARTIFICIAL_CONTROL_FAIL'):
            p.pool.map_owned(queue.pending(),callback,cpus=sorted(os.sched_getaffinity(0))[:2],
                limits={'max_workers':2},directory=self.root/'failed',deadline=math.inf,
                ready=queue.ready,on_complete=lambda t,h:queue.finish(t))
        receipt=c.read(self.root/'failed/parallel_receipt.json')
        self.assertTrue(receipt['all_reaped']);self.assertEqual(len(receipt['workers']),2)
        self.assertTrue(all(not Path('/proc',str(h['pid'])).exists() for h in receipt['workers']))
    def test_unsatisfied_dependencies_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError,'UNSATISFIED_TASK_DEPENDENCY'):
            p.pool.map_owned([self.tasks[1]],lambda t,d:{},cpus=[min(os.sched_getaffinity(0))],
                limits={'max_workers':1},directory=self.root/'blocked',deadline=math.inf,ready=lambda t:False)
    def test_original_pool_interface_remains_usable(self):
        receipt=p.pool.map_owned([self.tasks[0]],lambda t,d:dict(ordinary=True),
            cpus=[min(os.sched_getaffinity(0))],limits={'max_workers':1},
            directory=self.root/'legacy',deadline=math.inf)
        self.assertEqual(receipt['status'],'PASS');self.assertTrue(receipt['all_reaped'])
    def test_acceptance_exception_does_not_release_dependent(self):
        def reject(t,h):raise RuntimeError('ARTIFICIAL_ACCEPTANCE_REJECTED')
        q=p.CandidateQueue(self.tasks)
        with self.assertRaisesRegex(RuntimeError,'ARTIFICIAL_ACCEPTANCE_REJECTED'):
            p.pool.map_owned(q.pending(),lambda t,d:{},cpus=[min(os.sched_getaffinity(0))],
                limits={'max_workers':1},directory=self.root/'rejected',deadline=math.inf,
                ready=q.ready,on_complete=reject)
        receipt=c.read(self.root/'rejected/parallel_receipt.json')
        self.assertTrue(receipt['all_reaped']);self.assertEqual(len(receipt['workers']),1)


if __name__=='__main__':unittest.main()
