"""Bounded pressure diagnostics and saved run04 proof metadata; no science."""
import copy,json,unittest
from trottertracks.resource_applicability.h4_geometry import observer as obs,stopped_attempt_receipt as proof
from trottertracks.resource_applicability.h4_geometry.identity import Stop

def memory(psi=0):
    return {'available':400*2**30,'host_available':420*2**30,'observed_at':1.,'psi_full_avg10':psi,'psi_full_by_scope':{'host':0,'/sys/fs/cgroup/user.slice':psi,'/sys/fs/cgroup/user.slice/user-30038.slice':0},'oom_events':{'/sys/fs/cgroup/user.slice':1}}

def process(pid):return {'pid':pid,'parent':1,'uid':30038,'start':'123','rss':256*2**20,'address_space':512*2**20}

class PressureDiagnosticTests(unittest.TestCase):
    def test_pressure_stop_preserves_scope_values_and_first_reason(self):
        baseline=memory();state=obs.ObservationState(baseline,0.,workers=12)
        actual=memory(.02);record=state.evaluate(actual,[process(i) for i in range(10,23)],{'pid':24,'rss':20*2**20,'address_space':32*2**20},1.,1.01)
        self.assertEqual(state.first_failure['reason'],'memory_pressure')
        self.assertEqual(record['memory']['psi_full_by_scope']['/sys/fs/cgroup/user.slice'],.02)
        actual['psi_full_by_scope']['/sys/fs/cgroup/user.slice']=0
        self.assertEqual(record['memory']['psi_full_by_scope']['/sys/fs/cgroup/user.slice'],.02)
        first=copy.deepcopy(state.first_failure)
        state.evaluate(memory(),[process(10)],{'pid':24,'rss':20*2**20,'address_space':32*2**20},2.,2.01)
        self.assertEqual(state.first_failure,first)
        data=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
        self.assertLessEqual(len(data),obs.FRAME_CAP)
    def test_headroom_and_oom_gates_still_fail_closed(self):
        for key,value,reason in [('available',15*2**30,'memory_headroom'),('oom_events',{'changed':1},'oom_or_changed_cgroup')]:
            state=obs.ObservationState(memory(),0.,workers=12);m=memory();m[key]=value
            state.evaluate(m,[process(10)],{'pid':24,'rss':20*2**20,'address_space':32*2**20},1.,1.01)
            self.assertEqual(state.first_failure['reason'],reason)
    def test_normal_zero_pressure_has_no_failure(self):
        state=obs.ObservationState(memory(),0.,workers=12)
        row=state.evaluate(memory(),[process(10)],{'pid':24,'rss':20*2**20,'address_space':32*2**20},1.,1.01)
        self.assertIsNone(row['first_failure']);self.assertEqual(row['memory']['psi_full_avg10'],0)
    def test_recorded_run04_metadata(self):
        value=json.loads((proof.BASE/'RUNTIME_STOP_RECEIPT_v4.json').read_bytes())
        self.assertEqual(len(proof.validate_metadata(value)),14)
    def test_run04_original_cost_and_native_tamper_rejected(self):
        value=json.loads((proof.BASE/'RUNTIME_STOP_RECEIPT_v4.json').read_bytes())
        for key,new in [('cumulative_charged_bytes',0),('new_reserved_invocations',0),('driver_exit_exact_time',0),('all_owned_processes_ended',False)]:
            bad=copy.deepcopy(value);bad[key]=new
            with self.assertRaises(Stop):proof.validate_metadata(bad)
        bad=copy.deepcopy(value);bad['native_identity_audits'][0]['identities'][0]['classification']='SAME_OWNED_LIVE'
        with self.assertRaises(Stop):proof.validate_metadata(bad)
