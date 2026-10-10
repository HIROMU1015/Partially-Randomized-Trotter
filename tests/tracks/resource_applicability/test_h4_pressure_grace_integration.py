"""Only the newly approved pressure wiring and latest stop proof, pure fixtures."""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
from trottertracks.resource_applicability.h4_geometry import pressure_policy as policy,pressure_grace as grace,observer,launch_binding as bind,run08_receipt as proof
from trottertracks.resource_applicability.h4_geometry.identity import Stop,fingerprint
from trottertracks.resource_applicability.h4_geometry.resources import INITIAL_CGROUP_NS_INO

ROOT=Path(__file__).absolute().parents[3]
SCOPES=['/sys/fs/cgroup/user.slice','/sys/fs/cgroup/user.slice/user-30038.slice','/sys/fs/cgroup/user.slice/user-30038.slice/session-artificial.scope']


def memory(host=0.,now=0.,available=400*2**30):
    return dict(available=available,host_available=available,observed_at=now,psi_full_avg10=host,
        psi_full_by_scope={'host':host,**dict.fromkeys(SCOPES,0.0)},oom_events=dict.fromkeys(SCOPES,0),
        cgroup_namespace='cgroup:[%d]'%INITIAL_CGROUP_NS_INO,
        hierarchy=[dict(path='/sys/fs/cgroup',v2=True,is_root=True),*[dict(path=p,v2=True,is_root=False) for p in SCOPES]])


def role(pid):return dict(pid=pid,start=str(pid),parent=1,uid=30038,rss=32*2**20,address_space=64*2**20)


def documents():
    b=ROOT/'artifacts/resource_applicability/track_a_h4_production_run08/2026-10-10'
    p,a,r=[json.loads((b/name).read_bytes()) for name in ('plan_authorized_v18.json','authorization_v18.json','review_v18.json')]
    for d in (p,a,r):d['run_id']=bind.RUN_ID
    p.update(output_root='/home/AbeHiromu/artificial/output/'+bind.RUN_ID,control_root='/home/AbeHiromu/artificial/control/'+bind.RUN_ID,
             memory_pressure_profile=dict(path='artificial-policy.json',sha256='1'*64))
    a['pressure_amendment']=dict(approved=True,from_host_stop_percent=1.0,to_host_stop_percent=5.0,
        profile=p['memory_pressure_profile'],authority=dict(path='artificial-authority.json',sha256='2'*64))
    rebound(p,a,r);return p,a,r


def rebound(p,a,r):
    a['plan_fingerprint']=r['plan_fingerprint']=fingerprint('h4-newhost-plan-v2',p)
    r['authorization_digest']=fingerprint('h4-newhost-authorization-v2',a)


def observation(m):
    return dict(observed_monotonic=m['observed_at'],memory=m,scheduler_affinity=list(range(32)),online_cpus=list(range(32)),
        topology=[dict(cpu=c,package=0,core=c,busy_fraction=0.) for c in range(32)],
        filesystem=dict(block_bytes=4096,available_bytes=400*2**30,available_inodes=1000000),quota=dict(status='KNOWN',items=[]))


class GraceIntegrationTests(unittest.TestCase):
    def test_both_closed_profiles_and_no_unapproved_parameter_choices(self):
        self.assertEqual(policy.verify(policy.PARAMETERS),policy.PARAMETERS)
        self.assertEqual(policy.verify(policy.GRACE_PARAMETERS),policy.GRACE_PARAMETERS)
        for key,value in [('host_warning_continuous_seconds',60.),('host_full_avg10_stop_percent',10.),('host_only_min_available_bytes',0),('observation_interval_max_seconds',6)]:
            profile=dict(policy.GRACE_PARAMETERS);profile[key]=value
            with self.assertRaises(Stop):policy.verify(profile)
    def test_original1_percent_policy_still_stops_immediately(self):
        g=policy.PressureGuard(memory(),policy.PARAMETERS)
        self.assertEqual(g.decision(memory(1.26,1.),now=1.01)['reason'],'host_memory_pressure')
    def test_original_launch_baseline_does_not_fake_startup_timer_gap(self):
        g=policy.PressureGuard(memory(),policy.GRACE_PARAMETERS)
        value=g.decision(memory(1.26,20.),now=20.01)
        self.assertIsNone(value['reason']);self.assertEqual(value['warning_elapsed_seconds'],0.)
    def test_exact5_second_interval_and30_second_warning_boundary(self):
        g=policy.PressureGuard(memory(),policy.GRACE_PARAMETERS)
        for t in (1.,6.,11.,16.,21.,26.):self.assertIsNone(g.decision(memory(1.26,t),now=t)['reason'])
        self.assertEqual(g.decision(memory(1.26,31.),now=31.)['reason'],'host_memory_pressure_sustained')
    def test_nonzero_below1_percent_resets_warning(self):
        g=policy.PressureGuard(memory(),policy.GRACE_PARAMETERS)
        for t in range(1,25):g.decision(memory(1.26,float(t)),now=t+.01)
        self.assertFalse(g.decision(memory(.18,25.),now=25.01)['warning_active'])
        self.assertEqual(g.decision(memory(1.26,26.),now=26.01)['warning_elapsed_seconds'],0.)
    def test5_percent_and_first_failure_stay_immediate(self):
        g=policy.PressureGuard(memory(),policy.GRACE_PARAMETERS)
        self.assertEqual(g.decision(memory(5.,1.),now=1.01)['reason'],'host_memory_pressure_severe')
        self.assertEqual(g.decision(memory(0.,2.),now=2.01)['reason'],'host_memory_pressure_severe')
    def test_cgroup_oom_headroom_and_missing_data_do_not_get_grace(self):
        for mode in ('cgroup','oom','memory','missing'):
            g=policy.PressureGuard(memory(),policy.GRACE_PARAMETERS);g.decision(memory(1.26,1.),now=1.01)
            m=memory(1.26,2.)
            if mode=='cgroup':m['psi_full_by_scope'][SCOPES[0]]=.01
            elif mode=='oom':m['oom_events'][SCOPES[0]]=1
            elif mode=='memory':m['available']=int(152.25*2**30)-1
            else:m['psi_full_by_scope'].pop(SCOPES[0])
            self.assertIsNotNone(g.decision(m,now=2.01)['reason'])
    def test_observation_gap_greater_than5_seconds_stops(self):
        g=policy.PressureGuard(memory(),policy.GRACE_PARAMETERS);g.decision(memory(1.26,1.),now=1.01)
        self.assertIsNotNone(g.decision(memory(1.26,6.02),now=6.03)['reason'])
    def state(self):
        state=observer.ObservationState(memory(),0.,workers=4,pressure_profile=policy.GRACE_PARAMETERS,worker_cap=32*2**30,driver_pid=10)
        for pid in range(11,15):state.own_worker(pid)
        return state
    def test_observer_records_warning_without_suspending_observations(self):
        state=self.state();owners=[role(pid) for pid in range(10,15)]
        row=state.evaluate(memory(1.26,1.),owners,role(20),1.,1.01)
        self.assertIsNone(row['first_failure']);self.assertTrue(row['pressure_policy']['warning_active'])
        self.assertLessEqual(len((json.dumps(row,separators=(',',':'))+'\n').encode()),observer.FRAME_CAP)
        for t in range(2,31):self.assertIsNone(state.evaluate(memory(1.26,float(t)),owners,role(20),float(t),t+.01)['first_failure'])
        self.assertEqual(state.evaluate(memory(1.26,31.),owners,role(20),31.,31.01)['first_failure']['reason'],'host_memory_pressure_sustained')
    def test_driver8_worker32_and_observer_caps_still_stop_during_grace(self):
        for pid,field,cap in [(10,'address_space',8*2**30),(11,'rss',32*2**30),(20,'rss',64*2**20)]:
            state=self.state();owners=[role(i) for i in range(10,15)];sample=role(20)
            target=sample if pid==20 else next(r for r in owners if r['pid']==pid);target[field]=cap+1
            row=state.evaluate(memory(1.26,1.),owners,sample,1.,1.01)
            self.assertIn(row['first_failure']['reason'],('owned_role_rss_as','observer_rss_as'))
    def test_missing_ownership_is_not_excused_by_grace(self):
        row=self.state().evaluate(memory(1.26,1.),[role(10)],role(20),1.,1.01)
        self.assertEqual(row['first_failure']['reason'],'owned_memory_identity')
    def test_fresh_gate_below1_and152_point25_condition_preserved(self):
        p,a,r=documents();bind.fresh_gate(p,observation(memory(.18,1.)),now=1.01,pressure_profile=policy.GRACE_PARAMETERS)
        for m in (memory(1.26,1.),memory(0.,1.,int(152.25*2**30)-1)):
            with self.assertRaises(Stop):bind.fresh_gate(p,observation(m),now=1.01,pressure_profile=policy.GRACE_PARAMETERS)
    def test_closed_run09_amendment_requires_flags_binding_and_explicit_launch(self):
        p,a,r=documents();self.assertEqual(bind.authorize(p,a,r,explicit_launch=True).plan['run_id'],'h4-newhost-signal-compile-20261010-run09')
        for target in ('permission','review','explicit','threshold','pair'):
            p,a,r=documents()
            if target=='permission':a['pressure_amendment']['approved']=False
            elif target=='review':r['approved']=False
            elif target=='threshold':a['pressure_amendment']['to_host_stop_percent']=10.
            elif target=='pair':a['pressure_amendment']['from_host_stop_percent']=0.
            rebound(p,a,r)
            with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=target!='explicit')
    def test_production_evaluator_never_emits_launch_authorization(self):
        value=grace.GraceGuard(dict.fromkeys(SCOPES,0),SCOPES).evaluate(memory(1.26),now=0.)
        self.assertIs(value['runtime_authorization_emitted'],False)
    def test_run08_original_native_cost_unknown_exits_and17_byte_files(self):
        receipt=proof.verify_run08_stop(dict(newhost_run08_predecessor=dict(path=str(proof.RECEIPT),bytes=proof.RECEIPT_BYTES,sha256=proof.RECEIPT_SHA)))
        self.assertEqual(len(proof.validate_metadata(receipt)),6);self.assertEqual(len(receipt['files']),17)
        for key in ('charged_bytes','historical_driver_exit_code'):
            value=copy.deepcopy(receipt);value[key]=0
            with self.assertRaises(Stop):proof.validate_metadata(value)
    def test_run08_wrong_reference_and_same_owned_identity_still_stop(self):
        with patch.object(proof,'streaming_sha') as stream:
            with self.assertRaises(Stop):proof.verify_run08_stop(dict(newhost_run08_predecessor={}))
            stream.assert_not_called()
        identities=proof.validate_metadata(json.loads(proof.RECEIPT.read_bytes()))
        with patch('trottertracks.resource_applicability.h4_geometry.observer.process_sample',side_effect=lambda pid:identities[pid]):
            with self.assertRaises(Stop):proof.verify_run08_stop(dict(newhost_run08_predecessor=dict(path=str(proof.RECEIPT),bytes=proof.RECEIPT_BYTES,sha256=proof.RECEIPT_SHA)))
