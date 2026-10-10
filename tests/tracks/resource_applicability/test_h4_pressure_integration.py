"""Approved policy transport/gates, all process/science operations mocked."""
import copy,importlib.util,json,time,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch,Mock
from trottertracks.resource_applicability.h4_geometry import pressure_policy as pp,observer as obs,resources,launch_binding as bind,execution,run06_receipt
from trottertracks.resource_applicability.h4_geometry.identity import Stop,fingerprint
ROOT=Path(__file__).absolute().parents[3]
spec=importlib.util.spec_from_file_location('pressure_test_fixtures',Path(__file__).with_name('test_h4_host_pressure_proposal.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def documents():
    b=ROOT/'artifacts/resource_applicability/track_a_h4_production_run06/2026-10-10'
    p,a,r=[json.loads((b/n).read_bytes()) for n in ('plan_authorized_v16.json','authorization_v16.json','review_v16.json')]
    for d in (p,a,r):d['run_id']=bind.RUN_ID
    p.update(memory_pressure_profile={'path':'artificial-profile.json','sha256':'1'*64},output_root='/home/AbeHiromu/artificial/output/'+bind.RUN_ID,control_root='/home/AbeHiromu/artificial/control/'+bind.RUN_ID)
    a['pressure_amendment']={'approved':True,'from_host_stop_percent':0.0,'to_host_stop_percent':1.0,'profile':dict(p['memory_pressure_profile']),'authority':{'path':'artificial-authority.json','sha256':'2'*64}}
    rebound(p,a,r);return p,a,r

def rebound(p,a,r):
    a['plan_fingerprint']=r['plan_fingerprint']=fingerprint('h4-newhost-plan-v2',p)
    r['authorization_digest']=fingerprint('h4-newhost-authorization-v2',a)

def observation(memory):
    return dict(observed_monotonic=memory['observed_at'],memory=memory,
        scheduler_affinity=list(range(64)),online_cpus=list(range(64)),
        topology=[dict(cpu=i,package=0,core=i,busy_fraction=0) for i in range(64)],
        filesystem=dict(block_bytes=4096,available_bytes=400*2**30,available_inodes=1000000),
        quota=dict(status='KNOWN',items=[dict(status='DISABLED')]))

class PressureIntegrationTests(unittest.TestCase):
    def test_closed_profile_rejects_parameter_changes(self):
        pp.verify(dict(pp.PARAMETERS))
        for key,value in [('host_full_avg10_stop_percent',2.0),('host_only_min_available_bytes',0),('observation_max_age_seconds',6),('nonroot_full_avg10_stop_percent',.01),('mode','unlimited')]:
            v=dict(pp.PARAMETERS);v[key]=value
            with self.assertRaises(Stop):pp.verify(v)
    def test_pressure_amendment_permission_and_exact_profile_required(self):
        p,a,r=documents();bind.authorize(p,a,r,explicit_launch=True)
        for key,value in [('approved',False),('approved',1),('from_host_stop_percent',0),('to_host_stop_percent',2.0),('profile',{'path':'other','sha256':'3'*64})]:
            p,a,r=documents();a['pressure_amendment'][key]=value;rebound(p,a,r)
            with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
    def test_false_review_and_missing_explicit_launch_remain_closed(self):
        p,a,r=documents()
        with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=False)
        r['approved']=False;rebound(p,a,r)
        with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
    def test_observer_allows_bounded_host_only_point18(self):
        state=obs.ObservationState(fixture.memory(),0.,workers=4,pressure_profile=pp.PARAMETERS)
        row=state.evaluate(fixture.memory(.18),[fixture.role(i) for i in range(10,15)],fixture.role(20),1.,1.01)
        self.assertIsNone(row['first_failure']);self.assertEqual(row['memory']['psi_full_avg10'],.18)
        self.assertTrue(row['pressure_policy']['host_only_exception'])
    def test_observer_nonroot_and_host_threshold_stop(self):
        for mode in ('host','cgroup'):
            value=fixture.memory(1. if mode=='host' else .18)
            if mode=='cgroup':value['psi_full_by_scope'][fixture.SCOPES[0]]=.01
            row=obs.ObservationState(fixture.memory(),0.,workers=4,pressure_profile=pp.PARAMETERS).evaluate(value,[fixture.role(10)],fixture.role(20),1.,1.01)
            self.assertIsNotNone(row['first_failure'])
    def test_original_driver_baseline_oom_delta_stops_new_observer(self):
        launch=fixture.memory();current=fixture.memory();current['oom_events'][fixture.SCOPES[0]]=1
        state=obs.ObservationState(current,0.,workers=4,pressure_profile=pp.PARAMETERS,pressure_baseline=launch)
        row=state.evaluate(current,[fixture.role(10)],fixture.role(20),1.,1.01)
        self.assertEqual(row['first_failure']['reason'],'oom_or_changed_cgroup')
    def test_trusted_scope_changes_and_missing_data_stop(self):
        guard=pp.PressureGuard(fixture.memory(),pp.PARAMETERS)
        for mode in ('missing','hierarchy','namespace'):
            value=fixture.memory(.18)
            if mode=='missing':value['psi_full_by_scope'].pop(fixture.SCOPES[0])
            elif mode=='hierarchy':value['hierarchy'].reverse()
            else:value['cgroup_namespace']='cgroup:[1]'
            self.assertIsNotNone(guard.decision(value,now=1.01)['reason'])
    def test_legacy_monitor_and_observer_still_use_strict_zero(self):
        row=obs.ObservationState(fixture.memory(),0.,workers=4).evaluate(fixture.memory(.18),[fixture.role(10)],fixture.role(20),1.,1.01)
        self.assertEqual(row['first_failure']['reason'],'memory_pressure')
        monitor=resources.Monitor(4,{**fixture.memory(),'observed_at':0},clock=lambda:0)
        with self.assertRaises(Stop):monitor.check(fixture.memory(.18),[0])
    def test_driver_monitor_uses_same_profile_and_oom_baseline(self):
        clock=[0.];monitor=resources.Monitor(4,{**fixture.memory(),'observed_at':0},clock=lambda:clock[0],pressure_profile=pp.PARAMETERS)
        clock[0]=1.01;monitor.check(fixture.memory(.18),[32*2**20]*5)
        value=fixture.memory(.18);value['oom_events'][fixture.SCOPES[0]]=1
        clock[0]=2.01;value['observed_at']=2.
        with self.assertRaises(Stop):monitor.check(value,[0])
    def test_fresh_gate_accepts_point18_and_rejects1_percent(self):
        p,a,r=documents();bind.fresh_gate(p,observation(fixture.memory(.18)),now=1.01,pressure_profile=pp.PARAMETERS)
        with self.assertRaises(Stop):bind.fresh_gate(p,observation(fixture.memory(1.)),now=1.01,pressure_profile=pp.PARAMETERS)
    def test_fresh_gate_keeps120_point25GiB_and_other_resource_guards(self):
        p,a,r=documents()
        with self.assertRaises(Stop):bind.fresh_gate(p,observation(fixture.memory(0,64*2**30)),now=1.01,pressure_profile=pp.PARAMETERS)
        o=observation(fixture.memory(.18));o['quota']['status']='UNKNOWN'
        with self.assertRaises(Stop):bind.fresh_gate(p,o,now=1.01,pressure_profile=pp.PARAMETERS)
    def test_secondary_startup_rejects_changed_oom_before_observer(self):
        p,a,r=documents();baseline=fixture.memory();baseline['observed_at']=time.monotonic();current=copy.deepcopy(baseline);current['oom_events'][fixture.SCOPES[0]]=1
        permit=replace(bind.authorize(p,a,r,explicit_launch=True),launch_observation=observation(baseline))
        with patch.object(execution,'observe_memory',return_value=current),patch('trottertracks.resource_applicability.h4_geometry.observer.IndependentObserver') as observer,patch('trottertracks.resource_applicability.h4_geometry.workers.OwnedPool') as pool:
            with self.assertRaises(Stop):execution.OwnedRun(permit,a)
            observer.assert_not_called();pool.assert_not_called()
    def test_secondary_startup_rejects_host1_percent_before_observer(self):
        p,a,r=documents();baseline=fixture.memory();baseline['observed_at']=time.monotonic();current=copy.deepcopy(baseline);current['psi_full_avg10']=current['psi_full_by_scope']['host']=1.
        permit=replace(bind.authorize(p,a,r,explicit_launch=True),launch_observation=observation(baseline))
        with patch.object(execution,'observe_memory',return_value=current),patch.object(bind,'read_pressure_profile',return_value=dict(pp.PARAMETERS)),patch('trottertracks.resource_applicability.h4_geometry.observer.IndependentObserver') as observer,patch('trottertracks.resource_applicability.h4_geometry.workers.OwnedPool') as pool:
            with self.assertRaises(Stop):execution.OwnedRun(permit,a)
            observer.assert_not_called();pool.assert_not_called()
    def test_secondary_startup_forwards_profile_and_original_baseline(self):
        p,a,r=documents();baseline=fixture.memory();baseline['observed_at']=time.monotonic();current=copy.deepcopy(baseline);current['psi_full_avg10']=current['psi_full_by_scope']['host']=.18
        permit=replace(bind.authorize(p,a,r,explicit_launch=True),launch_observation=observation(baseline))
        with patch.object(execution,'observe_memory',return_value=current),patch.object(bind,'read_pressure_profile',return_value=dict(pp.PARAMETERS)),patch.object(execution.os,'sched_getaffinity',return_value={16}),patch.object(execution,'limit_owned_address_space'),patch.object(execution,'require_inherited_address_space'),patch.object(execution.threading,'Thread'),patch('trottertracks.resource_applicability.h4_geometry.observer.IndependentObserver') as observer,patch('trottertracks.resource_applicability.h4_geometry.workers.OwnedPool') as pool:
            run=execution.OwnedRun(permit,a,prepared_budget=Mock(root=Path(p['output_root'])))
            self.assertEqual(observer.call_args.kwargs['pressure_profile'],pp.PARAMETERS)
            self.assertEqual(observer.call_args.kwargs['pressure_baseline'],baseline)
            self.assertEqual(observer.call_args.kwargs['workers'],4)
            pool.assert_called_once()
    def test_role_cap_missing_observation_and_five_seconds_remain_closed(self):
        for mode in ('AS','RSS','missing','deadline'):
            state=obs.ObservationState(fixture.memory(),0.,workers=4,pressure_profile=pp.PARAMETERS)
            owners=[fixture.role(10)];m=fixture.memory(.18);end=1.01
            if mode=='AS':owners[0]['address_space']=8*2**30+1
            elif mode=='RSS':owners[0]['rss']=8*2**30+1
            elif mode=='missing':m=None
            else:end=6.01
            self.assertIsNotNone(state.evaluate(m,owners,fixture.role(20),1.,end)['first_failure'])
    def test_first_stop_and_trace_frame_preserved_under_new_policy(self):
        state=obs.ObservationState(fixture.memory(),0.,workers=4,pressure_profile=pp.PARAMETERS)
        record=state.evaluate(fixture.memory(1.),[fixture.role(i) for i in range(10,15)],fixture.role(20),1.,1.01);first=copy.deepcopy(state.first_failure)
        state.evaluate(fixture.memory(.18),[fixture.role(10)],fixture.role(20),2.,2.01)
        self.assertEqual(state.first_failure,first)
        self.assertLessEqual(len((json.dumps(record,separators=(',',':'))+'\n').encode()),obs.FRAME_CAP)
    def test_original_run06_stop_metadata_preserved(self):
        value=json.loads((run06_receipt.BASE/'RUNTIME_STOP_RECEIPT_v6.json').read_bytes())
        self.assertEqual(len(run06_receipt.validate_metadata(value)),6)
        bad=copy.deepcopy(value);bad['attempt_charged_bytes']=0
        with self.assertRaises(Stop):run06_receipt.validate_metadata(bad)
