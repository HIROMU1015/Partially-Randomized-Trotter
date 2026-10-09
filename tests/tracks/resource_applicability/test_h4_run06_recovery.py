"""Lower own concurrency, same limits and byte/native predecessor proof."""
import copy,importlib.util,json,unittest
from pathlib import Path
from trottertracks.resource_applicability.h4_geometry import launch_binding as bind,stopped_attempt_receipt as proof
from trottertracks.resource_applicability.h4_geometry.identity import Stop
ROOT=Path(__file__).absolute().parents[3]
spec=importlib.util.spec_from_file_location('run06_base_tests',Path(__file__).with_name('test_h4_run04_binding.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
class Run06RecoveryTests(unittest.TestCase):
    def test_four_workers_use_only_previously_allowed_cpu_roles(self):
        p,a,r=base.documents();p['requested_workers']=4;p['cpu_proposal']['workers']=p['cpu_proposal']['workers'][:4]
        a['allowed_cpus']=sorted(bind.roles(p));base.rebound(p,a,r)
        bind.authorize(p,a,r,explicit_launch=True)
        self.assertEqual(a['allowed_cpus'],[2,4,5,6,16,18]);self.assertEqual(p['caps']['worker_AS_RSS'],8*2**30)
    def test_reduced_workers_keep120_point25_GiB_admission(self):
        p,a,r=base.documents();p['requested_workers']=4;p['cpu_proposal']['workers']=p['cpu_proposal']['workers'][:4]
        a['allowed_cpus']=sorted(bind.roles(p));base.rebound(p,a,r)
        o={'observed_monotonic':1.,'memory':{'psi_full_avg10':0,'available':64*2**30,'observed_at':1.},'scheduler_affinity':list(range(64)),'online_cpus':list(range(64)),'topology':[{'cpu':i,'package':0,'core':i,'busy_fraction':0} for i in range(64)],'filesystem':{'block_bytes':4096,'available_bytes':400*2**30,'available_inodes':1000000},'quota':{'status':'KNOWN','items':[{'status':'DISABLED'}]}}
        with self.assertRaisesRegex(Stop,'admission'):bind.fresh_gate(p,o,now=1.1)
        o['memory']['available']=121*2**30;bind.fresh_gate(p,o,now=1.1)
    def test_run05_saved_host_pressure_stop(self):
        r=json.loads((proof.BASE/'RUNTIME_STOP_RECEIPT_v5.json').read_bytes());self.assertEqual(len(proof.validate_metadata(r)),14)
        self.assertEqual(r['first_stop']['memory']['psi_full_by_scope']['host'],.18)
    def test_run05_cost_and_scope_tamper_rejected(self):
        r=json.loads((proof.BASE/'RUNTIME_STOP_RECEIPT_v5.json').read_bytes())
        for key,value in [('attempt_charged_bytes',0),('new_actual_reservations',0),('all_owned_processes_ended',False),('exact_exit_time',0)]:
            bad=copy.deepcopy(r);bad[key]=value
            with self.assertRaises(Stop):proof.validate_metadata(bad)
        bad=copy.deepcopy(r);bad['first_stop']['memory']['psi_full_by_scope']['host']=0
        with self.assertRaises(Stop):proof.validate_metadata(bad)
