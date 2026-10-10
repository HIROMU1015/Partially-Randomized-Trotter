"""Pure scope/timer regressions. No science, runtime or extra processes."""
import copy,unittest
from trottertracks.resource_applicability.h4_geometry import pressure_grace_proposal as proposal,pressure_proposal as old
from trottertracks.resource_applicability.h4_geometry.resources import INITIAL_CGROUP_NS_INO

SCOPES=['/sys/fs/cgroup/user.slice','/sys/fs/cgroup/user.slice/user-30038.slice','/sys/fs/cgroup/user.slice/user-30038.slice/session-artificial.scope']


def memory(host=0,now=0.,available=400*2**30):
    return dict(available=available,host_available=available,observed_at=now,psi_full_avg10=host,
        psi_full_by_scope={'host':host,**dict.fromkeys(SCOPES,0.0)},
        oom_events=dict.fromkeys(SCOPES,0),cgroup_namespace='cgroup:[%d]'%INITIAL_CGROUP_NS_INO,
        hierarchy=[dict(path='/sys/fs/cgroup',v2=True,is_root=True),*[dict(path=p,v2=True,is_root=False) for p in SCOPES]])


def guard():return proposal.ProposedGraceGuard(dict.fromkeys(SCOPES,0),SCOPES)


class PressureGraceTests(unittest.TestCase):
    def test_recorded_run08_pressure_values_short_spike_get_grace(self):
        g=guard()
        for t,host in enumerate((0,.18,.32,.63,.88,1.26,0.)):
            value=g.evaluate(memory(host,float(t)),now=float(t))
            self.assertFalse(value['would_stop'])
        self.assertEqual(value['warning_elapsed_seconds'],0.)
    def test_continuous_warning_stops_at30_seconds(self):
        g=guard()
        for t in range(30):self.assertFalse(g.evaluate(memory(1.26,float(t)),now=float(t))['would_stop'])
        value=g.evaluate(memory(1.26,30.),now=30.)
        self.assertEqual(value['reason'],'host_memory_pressure_sustained')
    def test_warning_boundary1_and_immediate_boundary5(self):
        self.assertTrue(guard().evaluate(memory(1.),now=0.)['warning_active'])
        self.assertFalse(guard().evaluate(memory(.999),now=0.)['warning_active'])
        for host in (5.,5.01,100.):
            self.assertEqual(guard().evaluate(memory(host),now=0.)['reason'],'host_memory_pressure_severe')
    def test_zero_recovery_resets_timer(self):
        g=guard()
        for t in range(25):g.evaluate(memory(1.26,float(t)),now=float(t))
        self.assertFalse(g.evaluate(memory(0.,25.),now=25.)['would_stop'])
        for t in range(26,55):self.assertFalse(g.evaluate(memory(1.26,float(t)),now=float(t))['would_stop'])
        self.assertEqual(g.evaluate(memory(1.26,56.),now=56.)['reason'],'host_memory_pressure_sustained')
    def test_nonroot_each_scope_stops_immediately_during_grace(self):
        for path in SCOPES:
            g=guard();g.evaluate(memory(1.26),now=0.);value=memory(1.26,1.)
            value['psi_full_by_scope'][path]=.01
            self.assertEqual(g.evaluate(value,now=1.)['reason'],'nonroot_memory_pressure')
    def test_oom_or_effective_headroom_loss_stops_immediately(self):
        value=memory(1.26);value['oom_events'][SCOPES[0]]=1
        self.assertEqual(guard().evaluate(value,now=0.)['reason'],'oom_or_changed_cgroup')
        minimum=proposal.PARAMETERS['host_only_min_available_bytes']
        for host in (.18,1.26):
            self.assertEqual(guard().evaluate(memory(host,available=minimum-1),now=0.)['reason'],'host_pressure_insufficient_effective_headroom')
            self.assertFalse(guard().evaluate(memory(host,available=minimum),now=0.)['would_stop'])
    def test_zero_host_still_requires16GiB_headroom(self):
        self.assertEqual(guard().evaluate(memory(available=16*2**30-1),now=0.)['reason'],'memory_headroom')
    def test_no_observation_gap_is_accepted_as_grace(self):
        for now in (5.01,-1.,0.,float('nan'),float('inf'),True):
            g=guard();g.evaluate(memory(1.26),now=0.)
            self.assertTrue(g.evaluate(memory(1.26),now=now)['would_stop'])
    def test_stale_or_future_sample_stops(self):
        for now in (5.01,-.01):self.assertTrue(guard().evaluate(memory(),now=now)['would_stop'])
    def test_missing_scopes_oom_or_namespace_stop(self):
        for mode in ('scope','oom','namespace'):
            value=memory(1.26)
            if mode=='scope':value['psi_full_by_scope'].pop(SCOPES[0])
            elif mode=='oom':value['oom_events'].pop(SCOPES[0])
            else:value['cgroup_namespace']='cgroup:[1]'
            self.assertTrue(guard().evaluate(value,now=0.)['would_stop'])
    def test_hierarchy_order_or_aggregate_change_stops(self):
        g=guard();g.evaluate(memory(1.26),now=0.);value=memory(1.26,1.);value['hierarchy'].reverse()
        self.assertTrue(g.evaluate(value,now=1.)['would_stop'])
        value=memory(1.26);value['psi_full_avg10']=0
        self.assertTrue(guard().evaluate(value,now=0.)['would_stop'])
    def test_invalid_pressure_values_stop(self):
        for host in (float('nan'),float('inf'),-1.,101.,False,'1.26'):
            self.assertTrue(guard().evaluate(memory(host),now=0.)['would_stop'])
    def test_first_failure_latched_and_input_unchanged(self):
        g=guard();value=memory(5.);before=copy.deepcopy(value)
        self.assertEqual(g.evaluate(value,now=0.)['reason'],'host_memory_pressure_severe')
        self.assertEqual(g.evaluate(memory(0.,1.),now=1.)['reason'],'host_memory_pressure_severe')
        self.assertEqual(value,before)
    def test_proposal_unapproved_and_old_predicate_unchanged(self):
        result=guard().evaluate(memory(1.26),now=0.)
        for key in ('approved','runtime_authorization','production_wiring_present'):
            self.assertIs(result[key],False);self.assertIs(proposal.PARAMETERS[key],False)
        self.assertEqual(old.evaluate(memory(1.26),dict.fromkeys(SCOPES,0),SCOPES,now=0.)['reason'],'host_memory_pressure')
