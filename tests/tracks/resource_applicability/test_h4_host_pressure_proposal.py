"""Pure inactive proposal tests. No science, live observers, children or affinity."""
import copy,json,unittest
from trottertracks.resource_applicability.h4_geometry import pressure_proposal as proposal,observer as obs
from trottertracks.resource_applicability.h4_geometry.resources import INITIAL_CGROUP_NS_INO
ROOT='/sys/fs/cgroup'
SCOPES=[ROOT+'/user.slice',ROOT+'/user.slice/user-30038.slice',ROOT+'/user.slice/user-30038.slice/session-105501.scope']

def memory(host=0,available=400*2**30):
    return dict(available=available,host_available=available,observed_at=1.,psi_full_avg10=host,
        psi_full_by_scope={'host':host,**dict.fromkeys(SCOPES,0.0)},
        oom_events=dict.fromkeys(SCOPES,0),cgroup_namespace='cgroup:[%d]'%INITIAL_CGROUP_NS_INO,
        hierarchy=[dict(path=ROOT,v2=True,is_root=True),*[dict(path=p,v2=True,is_root=False) for p in SCOPES]])

def evaluate(value,now=1.01):return proposal.evaluate(value,dict.fromkeys(SCOPES,0),SCOPES,now=now)
def role(pid):return dict(pid=pid,start=str(pid),parent=1,uid=30038,rss=32*2**20,address_space=64*2**20)

class HostPressureProposalTests(unittest.TestCase):
    def test_recorded_stop_would_pass_only_pressure_proposal(self):
        from pathlib import Path
        raw=Path('/home/AbeHiromu/projects/h4-handoff-evidence/20261010/h4-production-run06-20261010/RUNTIME_STOP_RECEIPT_v6.json')
        receipt=json.loads(raw.read_bytes());stop=receipt['first_stop'];value=copy.deepcopy(stop['memory'])
        value.update(observed_at=stop['observation_started'],cgroup_namespace='cgroup:[%d]'%INITIAL_CGROUP_NS_INO,
                     hierarchy=memory()['hierarchy'])
        # These hierarchy entries are synthetic baseline metadata for replay;
        # no historical namespace/cgroup observation is inferred from this test.
        result=proposal.evaluate(value,value['oom_events'],SCOPES,now=stop['monotonic'])
        self.assertFalse(result['would_stop']);self.assertEqual(result['host_full_avg10'],.18)
        self.assertFalse(result['runtime_authorization'])
    def test_zero_and_small_host_values_with_full_headroom(self):
        for host in (0,.01,.18,.99):self.assertFalse(evaluate(memory(host))['would_stop'])
    def test_threshold_and_large_host_pressure_stop(self):
        for host in (1.,1.01,10.,100.):self.assertEqual(evaluate(memory(host))['reason'],'host_memory_pressure')
    def test_every_nonroot_scope_pressure_still_stops(self):
        for p in SCOPES:
            value=memory(.18);value['psi_full_by_scope'][p]=.01
            self.assertEqual(evaluate(value)['reason'],'nonroot_memory_pressure')
    def test_exception_requires120_point25_GiB_effective_available(self):
        self.assertTrue(evaluate(memory(.18,proposal.HOST_ONLY_MIN_AVAILABLE-1))['would_stop'])
        self.assertFalse(evaluate(memory(.18,proposal.HOST_ONLY_MIN_AVAILABLE))['would_stop'])
    def test_original16GiB_headroom_at_zero_host_pressure(self):
        self.assertEqual(evaluate(memory(0,16*2**30-1))['reason'],'memory_headroom')
        self.assertFalse(evaluate(memory(0,16*2**30))['would_stop'])
    def test_oom_increment_and_scope_change_stop(self):
        value=memory(.18);value['oom_events'][SCOPES[0]]=1
        self.assertEqual(evaluate(value)['reason'],'oom_or_changed_cgroup')
        value=memory();value['oom_events'].pop(SCOPES[0]);self.assertTrue(evaluate(value)['would_stop'])
    def test_missing_extra_or_only_host_pressure_scope_stop(self):
        for mode in ('missing','extra','only-host'):
            value=memory(.18)
            if mode=='missing':value['psi_full_by_scope'].pop(SCOPES[0])
            elif mode=='extra':value['psi_full_by_scope']['foreign']=0
            else:value['psi_full_by_scope']={'host':.18}
            self.assertTrue(evaluate(value)['would_stop'])
    def test_nonfinite_negative_boolean_or_excessive_values_stop(self):
        for v in (float('nan'),float('inf'),-.01,100.01,False,'0.18'):
            value=memory();value['psi_full_by_scope']['host']=v;value['psi_full_avg10']=v
            self.assertTrue(evaluate(value)['would_stop'])
    def test_stale_future_or_missing_observation_stop(self):
        for now in (6.01,.99,float('nan')):self.assertTrue(evaluate(memory(),now)['would_stop'])
        value=memory();value.pop('observed_at');self.assertTrue(evaluate(value)['would_stop'])
    def test_private_namespace_v1_hidden_or_duplicate_hierarchy_stop(self):
        for mode in ('namespace','v1','missing','duplicate','root'):
            value=memory(.18)
            if mode=='namespace':value['cgroup_namespace']='cgroup:[1]'
            elif mode=='v1':value['hierarchy'][1]['v2']=False
            elif mode=='missing':value['hierarchy'].pop()
            elif mode=='duplicate':value['hierarchy'].append(dict(value['hierarchy'][1]))
            else:value['hierarchy'][0]['path']='/hidden'
            self.assertTrue(evaluate(value)['would_stop'])
    def test_inconsistent_aggregate_or_invalid_available_stop(self):
        value=memory(.18);value['psi_full_avg10']=0;self.assertTrue(evaluate(value)['would_stop'])
        for available in (False,-1,1.0):self.assertTrue(evaluate(memory(0,available))['would_stop'])
    def test_existing_production_observer_still_stops_on_point18(self):
        state=obs.ObservationState(memory(),0.,workers=4)
        result=state.evaluate(memory(.18),[role(i) for i in range(10,15)],role(20),1.,1.01)
        self.assertEqual(result['first_failure']['reason'],'memory_pressure')
    def test_existing_role_AS_RSS_and_deadline_gates_preserved(self):
        for field in ('rss','address_space'):
            owners=[role(10)];owners[0][field]=8*2**30+1
            r=obs.ObservationState(memory(),0.,workers=4).evaluate(memory(),owners,role(20),1.,1.01)
            self.assertEqual(r['first_failure']['reason'],'owned_role_rss_as')
        r=obs.ObservationState(memory(),0.,workers=4).evaluate(memory(),[role(10)],role(20),1.,6.01)
        self.assertEqual(r['first_failure']['reason'],'observation_duration')
    def test_first_stop_cause_and_frame_budget_preserved(self):
        state=obs.ObservationState(memory(),0.,workers=4)
        record=state.evaluate(memory(.18),[role(i) for i in range(10,15)],role(20),1.,1.01)
        first=copy.deepcopy(state.first_failure)
        state.evaluate(memory(),[role(10)],role(20),2.,2.01)
        self.assertEqual(state.first_failure,first)
        self.assertLessEqual(len((json.dumps(record,separators=(',',':'))+'\n').encode()),obs.FRAME_CAP)
    def test_proposal_is_unapproved_and_input_is_not_mutated(self):
        self.assertFalse(proposal.POLICY['approved']);self.assertFalse(proposal.POLICY['runtime_authorization'])
        self.assertFalse(proposal.POLICY['production_wiring_present'])
        value=memory(.18);before=copy.deepcopy(value);result=evaluate(value)
        self.assertEqual(value,before);self.assertFalse(result['approved'])
