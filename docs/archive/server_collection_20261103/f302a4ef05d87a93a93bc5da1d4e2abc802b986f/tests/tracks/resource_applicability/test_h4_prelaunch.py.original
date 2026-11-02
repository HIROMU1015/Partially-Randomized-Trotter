"""Pure binding and bounded local fixtures; at most four minimal processes."""
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
from trottertracks.resource_applicability.h4_geometry import launch_binding as bind,prelaunch_audit as audit,resources,execution,observer
from trottertracks.resource_applicability.h4_geometry.identity import Stop,fingerprint

ROOT=Path(__file__).absolute().parents[3]
EVIDENCE=Path(os.environ['H4_PRELAUNCH_TEST_EVIDENCE'])
CONTRACT=json.loads((ROOT/'artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/zero_compute_plan_v2.json').read_text())
OLD=json.loads((ROOT/'artifacts/resource_applicability/track_a_h4_new_server_monitor_fix_a/2026-10-07/binding_draft_v1.json').read_text())


def documents(case='case'):
    root=EVIDENCE/case
    root.mkdir(exist_ok=True)
    plan=dict(schema_version=bind.VERSIONS['plan'],stage='signal_compile',run_id=bind.RUN_ID,
       source_commit='a'*40,source_root=str(root),source_hashes={'a.py':'a'*64},
       source_audit={'path':'audit.json','sha256':'a'*64},environment_profile={'path':'env.json','sha256':'a'*64},
       compiler_profile={'path':'compiler.json','sha256':'a'*64},stop_evidence_receipt={'path':'stop.json','sha256':'a'*64},
       input_root=str(root/'inputs'),stop_evidence_root=str(root/'stop'),output_root=str(root/'output'/bind.RUN_ID),
       control_root=str(root/'control'/bind.RUN_ID),inputs=OLD['expected_inputs']['expected_npz'],
       generation_freeze_digest=OLD['expected_inputs']['expected_generation_freeze_fingerprint'],
       templates=CONTRACT['templates'],contract_plan_fingerprint=bind.gates.PLAN_FP,
       compiler_fingerprint='b'*64,environment_fingerprint='c'*64,requested_workers=12,
       cpu_proposal=dict(driver=[13],workers=[[i] for i in range(1,13)],observer=[14]),
       carry=dict(bind.CARRY),caps=dict(actual_invocations=74804,wall_seconds=259200,output_bytes=10*2**30,
          driver_AS_RSS=8*2**30,worker_AS_RSS=8*2**30,headroom=16*2**30,monitor_seconds=5,observer_AS=256*2**20,observer_RSS=64*2**20),
       storage=audit.storage_projection(),sealed=True)
    auth=dict(schema_version=bind.VERSIONS['authorization'],stage='signal_compile',run_id=bind.RUN_ID,source_commit=plan['source_commit'],
       plan_fingerprint='',approved=True,runtime_authorization=True,allowed_cpus=list(range(1,15)),one_shot=True,
       permission='signal_compile',result_prior=True,environment_accepted=True,
       observer_role=dict(approved=True,runtime_authorization=True,AS_bytes=256*2**20,RSS_bytes=64*2**20),
       budget_amendment=dict(approved=True,**{'from':74784,'to':74804},authority_reference='SYNTHETIC_MEMORY_ONLY'))
    review=dict(schema_version=bind.VERSIONS['review'],stage='signal_compile',run_id=bind.RUN_ID,source_commit=plan['source_commit'],
       plan_fingerprint='',authorization_digest='',approved=True,runtime_authorization=True,reviewer='SYNTHETIC_FIXTURE',mandatory_stop=True)
    rebind(plan,auth,review);return plan,auth,review


def rebind(p,a,r):
    a['plan_fingerprint']=r['plan_fingerprint']=fingerprint('h4-newhost-plan-v2',p)
    r['authorization_digest']=fingerprint('h4-newhost-authorization-v2',a)


def observed(p):
    return dict(observed_monotonic=1.,memory=dict(psi_full_avg10=0,available=121*2**30,observed_at=1.),
       scheduler_affinity=list(range(64)),online_cpus=list(range(64)),
       topology=[dict(cpu=i,package=0,core=i,busy_fraction=0) for i in range(64)],
       filesystem=dict(available_bytes=400*2**30,available_inodes=1000000,block_bytes=4096),
       quota=dict(status='KNOWN',items=[dict(status='DISABLED')]))


class BindingTests(unittest.TestCase):
    def test_positive_pure_fixture(self):
        p,a,r=documents('pure');permit=bind.authorize(p,a,r,explicit_launch=True)
        self.assertEqual(permit.authorization,a)

    def test_each_permission_false_denies_without_io_affinity_spawn(self):
        for target,key in [(0,'sealed'),(1,'approved'),(1,'runtime_authorization'),(1,'one_shot'),(1,'result_prior'),
                           (1,'environment_accepted'),(2,'approved'),(2,'runtime_authorization'),(2,'mandatory_stop')]:
            docs=list(documents('false-%d-%s'%(target,key)));docs[target][key]=False;rebind(*docs)
            with patch.object(bind,'verify_runtime') as io_,patch.object(bind.os,'sched_setaffinity') as affinity,patch.object(subprocess,'Popen') as spawn:
                with self.assertRaises(Stop):bind.launch(*docs,explicit_launch=True)
                io_.assert_not_called();affinity.assert_not_called();spawn.assert_not_called()

    def test_explicit_launch_false(self):
        with self.assertRaises(Stop):bind.authorize(*documents('explicit'),explicit_launch=False)

    def test_plan_auth_review_digest_tamper(self):
        for target,key in [(0,'environment_fingerprint'),(1,'plan_fingerprint'),(2,'authorization_digest')]:
            d=list(documents('digest-%d'%target));d[target][key]='f'*64
            with self.assertRaises(Stop):bind.authorize(*d,explicit_launch=True)

    def test_required_schema_and_strict_bool_int(self):
        for target,key,value in [(0,'requested_workers',True),(1,'approved',1),(2,'reviewer',None)]:
            d=list(documents('type-%d'%target));d[target][key]=value
            with self.assertRaises(Stop):bind.authorize(*d,explicit_launch=True)
        d=list(documents('extra'));d[1]['new_role']=True
        with self.assertRaises(Stop):bind.authorize(*d,explicit_launch=True)

    def test_carry_never_reset(self):
        for field in bind.CARRY:
            d=list(documents('carry-'+field));d[0]['carry'][field]=0;rebind(*d)
            with self.assertRaises(Stop):bind.authorize(*d,explicit_launch=True)

    def test_74784_cap_insufficient_without_guaranteed_reuse(self):
        d=list(documents('insufficient'));d[0]['caps']['actual_invocations']=74784;rebind(*d)
        with self.assertRaisesRegex(Stop,'74764 remaining'):bind.authorize(*d,explicit_launch=True)

    def test_amendment_must_be_explicit_and_exact_plus20(self):
        for key,value in [('approved',False),('from',0),('to',74900),('authority_reference','')]:
            d=list(documents('amend-'+key));d[1]['budget_amendment'][key]=value;rebind(*d)
            with self.assertRaises(Stop):bind.authorize(*d,explicit_launch=True)

    def test_other_caps_not_relaxed(self):
        for key in ('wall_seconds','output_bytes','headroom','monitor_seconds','driver_AS_RSS','observer_AS'):
            d=list(documents('cap-'+key));d[0]['caps'][key]+=1;rebind(*d)
            with self.assertRaises(Stop):bind.authorize(*d,explicit_launch=True)

    def test_cpu_proposal_not_permission_or_overlap(self):
        d=list(documents('emptycpu'));d[1]['allowed_cpus']=[];rebind(*d)
        with self.assertRaises(Stop):bind.authorize(*d,explicit_launch=True)
        d=list(documents('overlap'));d[0]['cpu_proposal']['observer']=[13];rebind(*d)
        with self.assertRaises(Stop):bind.authorize(*d,explicit_launch=True)

    def test_observer_role_cannot_inherit_old_permission(self):
        d=list(documents('observer-role'));d[1]['observer_role']['approved']=False;rebind(*d)
        with self.assertRaises(Stop):bind.authorize(*d,explicit_launch=True)

    def test_generation_and_outside_home_rejected(self):
        d=list(documents('generation'));d[0]['stage']='input_generation';rebind(*d)
        with self.assertRaises(Stop):bind.authorize(*d,explicit_launch=True)
        d=list(documents('outside'));d[0]['output_root']='/tmp/'+bind.RUN_ID;rebind(*d)
        with self.assertRaises(Stop):bind.authorize(*d,explicit_launch=True)

    def test_static_actual_calls_and_capacity_formats(self):
        counts=audit.static_invocations(CONTRACT['templates']);self.assertEqual(counts['cumulative_actual_worst_case'],74804)
        self.assertEqual(counts['guaranteed_cache_savings'],0)
        storage=audit.storage_projection();self.assertLess(storage['cumulative_charge_bound'],10*2**30)
        self.assertEqual(storage['temporary_publishers'],14)

    def test_fresh_launch_cpu_memory_load_quota_capacity(self):
        p,a,r=documents('fresh');obs=observed(p);bind.fresh_gate(p,obs,now=1.1)
        for field in ('memory','quota','filesystem'):
            bad=copy.deepcopy(obs)
            if field=='memory':bad[field]['available']=120*2**30
            if field=='quota':bad[field]['status']='UNKNOWN'
            if field=='filesystem':bad[field]['available_inodes']=1
            with self.assertRaises(Stop):bind.fresh_gate(p,bad,now=1.1)
        bad=copy.deepcopy(obs);bad['topology'][1]['busy_fraction']=0.9
        with self.assertRaises(Stop):bind.fresh_gate(p,bad,now=1.1)
        with self.assertRaises(Stop):bind.fresh_gate(p,obs,now=7)

    def test_one_shot_marker_preserved_on_repeat(self):
        p,a,r=documents('oneshot');permit=bind.authorize(p,a,r,explicit_launch=True)
        bind.claim_once(permit)
        with self.assertRaises(FileExistsError):bind.claim_once(permit)
        self.assertTrue((Path(p['control_root'])/'one-shot.json').is_file())

    def test_role_affinity_mock_only(self):
        p,a,r=documents('affinity');permit=bind.authorize(p,a,r,explicit_launch=True)
        with patch.object(bind.os,'sched_setaffinity') as setter,patch.object(bind.os,'sched_getaffinity',return_value={13}):
            bind.role_affinity(permit,'driver');setter.assert_called_once_with(0,{13})

    def test_full_launch_order_stub_science_and_no_real_affinity(self):
        p,a,r=documents('pipeline');Path(p['output_root']).parent.mkdir()
        order=[]
        with patch.object(bind,'verify_runtime',side_effect=lambda _: (order.append('source_profile') or (CONTRACT,{}))),\
             patch.object(bind,'verify_frozen_receipts',side_effect=lambda _:order.append('receipt')),\
             patch.object(bind,'host_readonly',return_value=observed(p)),\
             patch.object(bind,'fresh_gate',side_effect=lambda *_:order.append('fresh')),\
             patch.object(bind,'role_affinity',side_effect=lambda *_:order.append('CPU_MOCK')),\
             patch.object(bind,'install_write_guard'),\
             patch.object(execution,'signal_stage',side_effect=lambda *_a,**_kw:(order.append('SCIENCE_STUB') or {'status':'MAP_COMPLETE_STOP'})):
            result=bind.launch(p,a,r,explicit_launch=True)
        self.assertEqual(result['status'],'MAP_COMPLETE_STOP')
        self.assertEqual(order,['source_profile','receipt','fresh','CPU_MOCK','SCIENCE_STUB'])
        self.assertTrue((Path(p['output_root'])/'launch-stop.json').is_file())

    def test_runtime_rejects_changed_source_profile_before_science(self):
        p,a,r=documents('profile-negative');permit=bind.authorize(p,a,r,explicit_launch=True)
        with patch.object(bind,'reference',side_effect=Stop('changed source/profile')),\
             patch.object(bind,'private_path',return_value=ROOT),patch.object(execution,'signal_stage') as science:
            with self.assertRaisesRegex(Stop,'changed source/profile'):bind.verify_runtime(permit)
            science.assert_not_called()

    def test_shared_worker_and_temp_writes_denied_but_reads_allowed(self):
        p,_,_=documents('write-guard')
        bind.write_guard(p,'open',('/usr/lib/anything.py','r',os.O_RDONLY))
        with self.assertRaises(Stop):bind.write_guard(p,'open',('/tmp/file','w',os.O_WRONLY))
        with self.assertRaises(Stop):bind.write_guard(p,'open',('/home/AbeHiromu/.shared-config','w',os.O_WRONLY))
        with self.assertRaises(Stop):bind.write_guard(p,'open',(p['control_root']+'/private-temp/file','w',os.O_WRONLY))
        with self.assertRaises(Stop):bind.write_guard(p,'open',(p['output_root']+'/record-x.json','w',os.O_WRONLY),worker=True)
        bind.write_guard(p,'open',(p['output_root']+'/record-x.json','w',os.O_WRONLY))
        with self.assertRaises(Stop):bind.write_guard(p,'open',('record-x.json','w',os.O_WRONLY))
        resources.MANAGED_WRITE_CONTEXT.root=Path(p['output_root'])
        try:bind.write_guard(p,'open',('record-x.json.pending','w',os.O_WRONLY))
        finally:resources.MANAGED_WRITE_CONTEXT.root=None


class AccountingReceiptTests(unittest.TestCase):
    def test_streaming_bytes_not_arrays_and_missing_mismatch_symlink(self):
        root=EVIDENCE/'byte-fixture';root.mkdir();(root/'input.bin').write_bytes(b'ARTIFICIAL\x00'*10000)
        actual=audit.streaming_sha(root/'input.bin')
        self.assertEqual(actual['sha256'],hashlib.sha256((root/'input.bin').read_bytes()).hexdigest())
        self.assertFalse(audit.receipt_inventory(root,{'absent.bin':'a'*64})['complete'])
        with self.assertRaises(Stop):audit.receipt_inventory(root,{'input.bin':'a'*64})
        (root/'link.bin').symlink_to(root/'input.bin')
        with self.assertRaises(Stop):audit.streaming_sha(root/'link.bin')

    def test_cached_journal_charges_carry_and_no_full_reread(self):
        root=EVIDENCE/'budget-fixture';b=resources.OutputBudget(root,prior_charge=165214360,file_limits=bind.FILE_LIMITS)
        try:
            b.write('record-first.json',b'{}')
            with patch.object(resources.os,'read',side_effect=AssertionError('whole journal reread')):
                b.write('record-second.json',b'{}')
            self.assertEqual(b.cached_charge,165214360+128+2*(4+128))
            with self.assertRaises(Stop):b.write('record-over.json',b'x'*4097)
            with (root/'byte-budget.journal').open('ab') as f:f.write(b'0'*128)
            with self.assertRaises(Stop):b.reserve(1)
        finally:b.close()

    def test_partial_reservation_stops_before_publication(self):
        root=EVIDENCE/'partial-journal';b=resources.OutputBudget(root)
        try:
            with patch.object(resources.os,'write',return_value=1):
                with self.assertRaises(Stop):b.write('record-x.json',b'{}')
            self.assertFalse((root/'record-x.json').exists())
        finally:b.close()

    def test_bounded_control_log(self):
        path=EVIDENCE/'log-fixture.txt'
        with bind.CappedDriverLog(path,cap=4) as log:
            log.write('abcd')
            with self.assertRaises(Stop):log.write('e')
        self.assertEqual(path.read_bytes(),b'abcd')

    def test_watch_cleanup_even_if_control_log_full(self):
        run=execution.OwnedRun.__new__(execution.OwnedRun)
        run.finished=type('Event',(),{'wait':lambda *_:False})();run.failure=None
        run.monitor=type('Monitor',(),{'poll':lambda _:(_ for _ in ()).throw(Stop('first')),'stop_children':lambda _:None})()
        with patch('builtins.print',side_effect=Stop('log full')),patch.object(run.monitor,'stop_children') as stop,patch('_thread.interrupt_main') as interrupt:
            with self.assertRaisesRegex(Stop,'log full'):run._watch()
            self.assertEqual(str(run.failure),'first');stop.assert_called_once();interrupt.assert_called_once()


class MinimalCleanupTests(unittest.TestCase):
    def test_driver_exit_reparents_and_reaps_worker_and_observer(self):
        root=EVIDENCE/'minimal-family';root.mkdir()
        libc=ctypes.CDLL(None);self.assertEqual(libc.prctl(36,1,0,0,0),0)  # own test process subreaper only
        process=subprocess.Popen([sys.executable,'-P','-B',str(Path(__file__).with_name('h4_prelaunch_minimal_process.py')),'driver',str(root)],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True)
        owner=observer.OwnedIdentity(observer.identity(observer.process_sample(process.pid)),os.getpid());family=None
        try:
            until=time.monotonic()+8
            while not (root/'family.json').exists() and time.monotonic()<until:time.sleep(0.05)
            family=json.loads((root/'family.json').read_text());process.wait(timeout=8)
            for role in ('worker','observer'):
                pid=family[role]['pid'];deadline=time.monotonic()+8;reaped=False
                while time.monotonic()<deadline:
                    waited,status=os.waitpid(pid,os.WNOHANG)
                    if waited:reaped=True;break
                    time.sleep(0.05)
                self.assertTrue(reaped,role+' did not exit/reap')
            rows=[json.loads(x) for x in (root/'orphan-observer.jsonl').read_text().splitlines()]
            self.assertTrue(any(r['kind']=='first_stop' for r in rows))
        finally:
            owner.terminate();owner.close()
            if process.poll() is None:process.wait(timeout=3)
            if family:
                for role in ('worker','observer'):
                    pid=family[role]['pid']
                    try:
                        current=observer.process_sample(pid)
                        if all(current[k]==family[role][k] for k in ('uid','pid','start')):
                            fd=os.pidfd_open(pid);signal.pidfd_send_signal(fd,signal.SIGKILL);os.close(fd)
                            os.waitpid(pid,0)
                    except (OSError,Stop):pass
            libc.prctl(36,0,0,0,0)
