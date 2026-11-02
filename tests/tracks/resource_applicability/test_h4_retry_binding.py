"""Pure run03 permission/carry regressions. No circuits, compile, or children."""
import copy
import json
import os
from pathlib import Path
import unittest
from trottertracks.resource_applicability.h4_geometry import launch_binding as bind,prelaunch_audit as audit,resources,retry_receipt
from trottertracks.resource_applicability.h4_geometry.identity import Stop,fingerprint

ROOT=Path(__file__).absolute().parents[3]
OLD=ROOT/'artifacts/resource_applicability/track_a_h4_approved_relaunch/2026-10-09'
EVIDENCE=Path(os.environ['H4_PRELAUNCH_TEST_EVIDENCE'])


def rebound(p,a,r):
    a['plan_fingerprint']=r['plan_fingerprint']=fingerprint('h4-newhost-plan-v2',p)
    r['authorization_digest']=fingerprint('h4-newhost-authorization-v2',a)


def documents():
    p=json.loads((OLD/'plan_authorized_v11.json').read_bytes())
    a=json.loads((OLD/'authorization_v11.json').read_bytes())
    r=json.loads((OLD/'review_v11.json').read_bytes())
    for d in (p,a,r):d['run_id']=bind.RUN_ID
    p.update(carry=dict(bind.CARRY),output_root=str(EVIDENCE/'output'/bind.RUN_ID),control_root=str(EVIDENCE/'control'/bind.RUN_ID))
    p['caps'].update(actual_invocations=74805,output_bytes=17*2**30)
    p['storage']=audit.storage_projection(library_cache_bytes=29816,output_cap=17*2**30)
    a['budget_amendment']=dict(approved=True,**{'from':74804,'to':74805},authority_reference='ARTIFICIAL_MEMORY_ONLY')
    a['output_budget_amendment']=dict(approved=True,**{'from':13*2**30,'to':17*2**30},authority_reference='ARTIFICIAL_MEMORY_ONLY')
    rebound(p,a,r);return p,a,r


class RetryBindingTests(unittest.TestCase):
    def test_current_carry_and_full_map_bound(self):
        p,a,r=documents();counts=audit.static_invocations(p['templates'],carry_actual=21)
        self.assertEqual(counts['logical_wrappers'],74784)
        self.assertEqual(counts['cumulative_actual_worst_case'],74805)
        self.assertEqual(counts['guaranteed_cache_savings'],0)
        self.assertEqual(bind.CARRY,dict(actual_invocations=21,charged_bytes=8692723164,wall_seconds=5766.582514658794))
        self.assertEqual(p['storage']['cumulative_charge_bound'],17429694796)
        self.assertEqual(p['storage']['remaining_charge_margin_bytes'],823916212)
        bind.authorize(p,a,r,explicit_launch=True)
        for invalid in (True,-1,21.0):
            with self.assertRaises(Stop):audit.static_invocations(p['templates'],carry_actual=invalid)

    def test_drafts_and_missing_explicit_launch_stay_closed(self):
        p,a,r=documents()
        with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=False)
        for target,key in [(p,'sealed'),(a,'approved'),(a,'runtime_authorization'),(r,'approved'),(r,'runtime_authorization')]:
            old=target[key];target[key]=False;rebound(p,a,r)
            with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
            target[key]=old

    def test_carry_is_never_reset_or_refunded(self):
        for key in bind.CARRY:
            p,a,r=documents();p['carry'][key]=bind.PRIOR_CARRY[key];rebound(p,a,r)
            with self.assertRaisesRegex(Stop,'carry'):bind.authorize(p,a,r,explicit_launch=True)

    def test_previous_actual_cap_cannot_cover_fresh_full_map(self):
        p,a,r=documents();p['caps']['actual_invocations']=74804
        a['budget_amendment'].update(**{'from':74784,'to':74804});rebound(p,a,r)
        with self.assertRaisesRegex(Stop,'remaining actual'):bind.authorize(p,a,r,explicit_launch=True)

    def test_both_amendments_require_exact_explicit_authority(self):
        for section in ('budget_amendment','output_budget_amendment'):
            for key,value in [('approved',False),('approved',1),('from',0),('to',999999),('authority_reference','')]:
                p,a,r=documents();a[section][key]=value;rebound(p,a,r)
                with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
        for key,value in [('output_bytes',18*2**30),('actual_invocations',74806),('monitor_seconds',6)]:
            p,a,r=documents();p['caps'][key]=value;rebound(p,a,r)
            with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)

    def test_previous_output_cap_still_fails_fresh_gate(self):
        p,a,r=documents();p['caps']['output_bytes']=13*2**30
        a['output_budget_amendment'].update(**{'from':10*2**30,'to':13*2**30});rebound(p,a,r)
        bind.authorize(p,a,r,explicit_launch=True)
        observation=dict(observed_monotonic=1.,memory=dict(psi_full_avg10=0,available=121*2**30,observed_at=1.),
            scheduler_affinity=list(range(64)),online_cpus=list(range(64)),
            topology=[dict(cpu=i,package=0,core=i,busy_fraction=0) for i in range(64)],
            filesystem=dict(block_bytes=4096,available_bytes=400*2**30,available_inodes=1000000),
            quota=dict(status='KNOWN',items=[dict(status='DISABLED')]))
        with self.assertRaisesRegex(Stop,'cumulative output'):bind.fresh_gate(p,observation,now=1.1)
        p,a,r=documents();bind.fresh_gate(p,observation,now=1.1)

    def test_native_stop_metadata_tamper_is_rejected(self):
        receipt=json.loads((retry_receipt.BASE/'RUNTIME_STOP_RECEIPT_v2.json').read_bytes())
        retry_receipt.validate_metadata(receipt,bind.PRIOR_CARRY,bind.CARRY)
        for key,value in [('run_id','foreign'),('source_commit','0'*40),('new_reserved_invocations',0),
                          ('completed_wrappers',1),('actual_compiler_start_confirmed',True),
                          ('cumulative_charged_bytes',0),('conservative_wall_upper_seconds',0),
                          ('driver_exit_exact_time',1),('all_owned_processes_ended',False)]:
            bad=copy.deepcopy(receipt);bad[key]=value
            with self.assertRaises(Stop):retry_receipt.validate_metadata(bad,bind.PRIOR_CARRY,bind.CARRY)
        for kind in ('remaining','mismatch','missing'):
            bad=copy.deepcopy(receipt)
            if kind=='remaining':bad['native_identity_audits'][0]['identities'][0]['classification']='SAME_OWNED_LIVE'
            if kind=='mismatch':bad['native_identity_audits'][1]['identities'][0]['expected']['start']='0'
            if kind=='missing':bad['native_identity_audits'][1]['identities'].pop()
            with self.assertRaises(Stop):retry_receipt.validate_metadata(bad,bind.PRIOR_CARRY,bind.CARRY)

    def test_runtime_budget_keeps_charge_on_failure(self):
        budget=resources.OutputBudget(EVIDENCE/'journal-fixture',cap=17*2**30,prior_charge=bind.CARRY['charged_bytes'])
        try:
            budget.reserve(128);before=budget.cached_charge
            with self.assertRaises(Stop):budget.reserve(17*2**30)
            self.assertEqual(budget.cached_charge,before)
            self.assertGreater(before,bind.CARRY['charged_bytes'])
        finally:budget.close()
