"""Pure run04 budget/recovery binding. No compile, arrays, children or affinity."""
import copy
import json
from pathlib import Path
import unittest
from trottertracks.resource_applicability.h4_geometry import launch_binding as bind,prelaunch_audit as audit,run03_receipt
from trottertracks.resource_applicability.h4_geometry.identity import Stop,fingerprint
ROOT=Path(__file__).absolute().parents[3]
OLD=ROOT/'artifacts/resource_applicability/track_a_h4_production_run03/2026-10-09-authorized'

def documents():
    p,a,r=[json.loads((OLD/name).read_bytes()) for name in ('plan_authorized_v13.json','authorization_v13.json','review_v13.json')]
    for d in (p,a,r):d['run_id']=bind.RUN_ID
    p.update(carry=dict(bind.CARRY),output_root='/home/AbeHiromu/artificial/output/'+bind.RUN_ID,control_root='/home/AbeHiromu/artificial/control/'+bind.RUN_ID)
    p['storage']=audit.storage_projection(library_cache_bytes=29816,prior_charge=0,output_cap=17*2**30)
    ref={'path':'artificial-authority.json','sha256':'1'*64}
    a['budget_accounting']={'scope':'per_attempt','exclude_failed_carry':True,'authority':dict(ref)}
    a['recovery_policy']={'approved':True,'mode':'investigate_fix_fresh_restart','scope':'signal_compile_only','authority':dict(ref)}
    rebound(p,a,r);return p,a,r

def rebound(p,a,r):
    a['plan_fingerprint']=r['plan_fingerprint']=fingerprint('h4-newhost-plan-v2',p)
    r['authorization_digest']=fingerprint('h4-newhost-authorization-v2',a)

class Run04BindingTests(unittest.TestCase):
    def test_authorized_metadata(self):
        p,a,r=documents();bind.authorize(p,a,r,explicit_launch=True)
        self.assertEqual(p['carry'],dict(actual_invocations=0,charged_bytes=0,wall_seconds=0.0))
    def test_full_map_without_past_budget(self):
        p,a,r=documents();counts=audit.static_invocations(p['templates'],carry_actual=0)
        self.assertEqual(counts['cumulative_actual_worst_case'],74784)
        self.assertEqual(p['storage']['cumulative_charge_bound'],8736971632)
        self.assertGreater(p['storage']['remaining_charge_margin_bytes'],0)
    def test_any_nonzero_or_boolean_carry_rejected(self):
        for key,value in [('actual_invocations',22),('charged_bytes',12956511264),('wall_seconds',6004.111340102032),('actual_invocations',False),('wall_seconds',0)]:
            p,a,r=documents();p['carry'][key]=value;rebound(p,a,r)
            with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
    def test_accounting_permission_rejected(self):
        for key,value in [('scope','cumulative'),('exclude_failed_carry',False),('exclude_failed_carry',1)]:
            p,a,r=documents();a['budget_accounting'][key]=value;rebound(p,a,r)
            with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
    def test_recovery_scope_rejected(self):
        for key,value in [('approved',False),('approved',1),('mode','blind_retry'),('scope','next_stage')]:
            p,a,r=documents();a['recovery_policy'][key]=value;rebound(p,a,r)
            with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
    def test_authority_reference_rejected(self):
        for section in ('budget_accounting','recovery_policy'):
            for key,value in [('path','/tmp/authority.json'),('path','../authority.json'),('path',''),('sha256','invalid')]:
                p,a,r=documents();a[section]['authority'][key]=value;rebound(p,a,r)
                with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
    def test_runtime_false_and_missing_explicit_launch_rejected(self):
        p,a,r=documents()
        with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=False)
        for section,key in [('plan','sealed'),('authorization','approved'),('authorization','runtime_authorization'),('review','approved'),('review','runtime_authorization')]:
            p,a,r=documents();{'plan':p,'authorization':a,'review':r}[section][key]=False;rebound(p,a,r)
            with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
    def test_cpu_overlap_rejected(self):
        p,a,r=documents();p['cpu_proposal']['observer']=p['cpu_proposal']['driver'];rebound(p,a,r)
        with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
    def test_wrong_run_and_arbitrary_caps_rejected(self):
        for key,value in [('actual_invocations',74806),('output_bytes',21*2**30),('monitor_seconds',6)]:
            p,a,r=documents();p['caps'][key]=value;rebound(p,a,r)
            with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
        p,a,r=documents();p['run_id']='consumed-run';rebound(p,a,r)
        with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
    def test_saved_run03_metadata(self):
        receipt=json.loads((run03_receipt.BASE/'RUNTIME_STOP_RECEIPT_v3.json').read_bytes())
        run03_receipt.validate_metadata(receipt,bind.RUN02_CARRY,bind.HISTORICAL_CARRY)
        self.assertEqual(receipt['actual_invocations_consumed_or_reserved'],22)
    def test_saved_run03_tampering_rejected(self):
        receipt=json.loads((run03_receipt.BASE/'RUNTIME_STOP_RECEIPT_v3.json').read_bytes())
        for key,value in [('actual_invocations_consumed_or_reserved',0),('cumulative_charged_bytes',0),('driver_exit_exact_time',0),('actual_compiler_start_confirmed',False),('all_owned_processes_ended',False)]:
            bad=copy.deepcopy(receipt);bad[key]=value
            with self.assertRaises(Stop):run03_receipt.validate_metadata(bad,bind.RUN02_CARRY,bind.HISTORICAL_CARRY)
        bad=copy.deepcopy(receipt);bad['native_identity_audits'][0]['identities'][0]['classification']='SAME_OWNED_LIVE'
        with self.assertRaises(Stop):run03_receipt.validate_metadata(bad,bind.RUN02_CARRY,bind.HISTORICAL_CARRY)
