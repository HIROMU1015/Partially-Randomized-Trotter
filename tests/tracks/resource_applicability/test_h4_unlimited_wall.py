"""Only wall-removal wiring and existing resource STOPs; no real children."""
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trottertracks.resource_applicability.h4_geometry import wall_time as policy, observer, resources, launch_binding as lb
from trottertracks.resource_applicability.h4_geometry.identity import Stop, fingerprint

ROOT=Path(__file__).resolve().parents[3]
def module(name):
    s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
fx=module('test_h4_pressure_grace_integration')
family=module('test_h4_resume02_continuation')


def state(*,unlimited=True):
    s=observer.ObservationState(fx.memory(),0.,prior_wall=200*3600.,workers=4,pressure_profile=fx.policy.GRACE_PARAMETERS,
        worker_cap=32*2**30,driver_pid=10,wall_cap=None if unlimited else resources.WALL_CAP,wall_unlimited_authorized=unlimited)
    s.own_worker(11);return s


def evaluate(s,m=None,owners=None,obs=None,begun=1.,ended=1.01):
    return s.evaluate(fx.memory(now=begun) if m is None else m,
        [fx.role(10),fx.role(11)] if owners is None else owners,fx.role(20) if obs is None else obs,begun,ended)


class WallTests(unittest.TestCase):
    def test_driver_unlimited_keeps_cumulative_time_beyond72h(self):
        times=iter((0.,200*3600.));b=resources.WallBudget(35474.,cap=None,clock=lambda:next(times))
        self.assertEqual(b.consumed(),35474.+200*3600.)
    def test_driver_legacy_default_still_stops_and_invalid_time_fails(self):
        times=iter((0.,72*3600.+1));b=resources.WallBudget(0.,clock=lambda:next(times))
        with self.assertRaises(Stop):b.consumed()
        for bad in (float('nan'),float('inf'),-1):
            with self.assertRaises(Stop):resources.WallBudget(bad,cap=None)
        times=iter((0.,float('inf')));b=resources.WallBudget(0.,cap=None,clock=lambda:next(times))
        with self.assertRaises(Stop):b.consumed()
    def test_observer_unlimited_and_legacy_paths(self):
        self.assertIsNone(evaluate(state())['first_failure'])
        self.assertEqual(evaluate(state(unlimited=False))['first_failure']['reason'],'cumulative_wall')
        with self.assertRaises(Stop):observer.ObservationState(fx.memory(),0.,wall_cap=None)
    def test_five_second_deadline_interval_staleness_and_missing_still_stop(self):
        cases=[(None,6.,6.01),(fx.memory(now=0.),1.,6.1),(fx.memory(now=0.),6.,6.01)]
        for m,b,e in cases:
            self.assertIsNotNone(evaluate(state(),m=m,begun=b,ended=e)['first_failure'])
        s=state();self.assertEqual(s.evaluate(None,[fx.role(10),fx.role(11)],fx.role(20),1.,1.01)['first_failure']['reason'],'missing_observation')
    def test_driver_worker_observer_memory_limits_still_stop(self):
        for pid,field,cap in [(10,'rss',8*2**30),(11,'address_space',32*2**30),(20,'rss',64*2**20)]:
            owners=[fx.role(10),fx.role(11)];sample=fx.role(20)
            target=sample if pid==20 else next(r for r in owners if r['pid']==pid);target[field]=cap+1
            self.assertIn(evaluate(state(),owners=owners,obs=sample)['first_failure']['reason'],('owned_role_rss_as','observer_rss_as'))
    def test_pressure_oom_missing_and_first_stop_not_relaxed(self):
        for mode in ('pressure','oom','memory','missing'):
            s=state();m=fx.memory(now=1.)
            if mode=='pressure':m['psi_full_by_scope']['host']=m['psi_full_avg10']=5.
            elif mode=='oom':m['oom_events'][fx.SCOPES[0]]=1
            elif mode=='memory':m['available']=resources.HEADROOM-1
            else:m['psi_full_by_scope'].pop(fx.SCOPES[0])
            first=evaluate(s,m=m)['first_failure'];self.assertIsNotNone(first)
            self.assertEqual(evaluate(s,m=fx.memory(now=2.),begun=2.,ended=2.01)['first_failure'],first)
    def test_owned_identity_and_nonfinite_wall_still_fail_closed(self):
        s=state();self.assertEqual(evaluate(s,owners=[fx.role(10),fx.role(12)])['first_failure']['reason'],'owned_memory_identity')
        s=state();s.prior_wall=float('inf');self.assertEqual(evaluate(s)['first_failure']['reason'],'cumulative_wall')
    def test_trace_is_byte_bounded_with_terminal_reserve(self):
        trace=observer.Trace(999,observer.TERMINAL_RESERVE+32)
        with patch.object(observer.os,'write',side_effect=lambda fd,b:len(b)),patch.object(observer.os,'fsync'):
            trace.write({'x':1});trace.write({'x':1});trace.write({'x':1});trace.write({'x':1})
            with self.assertRaises(Stop):trace.write({'x':1})
            trace.write({'kind':'first_stop','reason':'observer output budget'},terminal=True)
        self.assertEqual(policy.TRACE_CAP,2*2**30);observer.Trace(999,policy.TRACE_CAP)
    def test_closed_policy_no_other_cap_or_retry_change(self):
        self.assertEqual(policy.verify(copy.deepcopy(policy.PARAMETERS)),policy.PARAMETERS)
        for k,v in [('observer_trace_bytes',3*2**30),('observation_deadline_seconds',10),('automatic_retry',True),('memory_pressure_oom_capacity_identity_stops',False)]:
            p=copy.deepcopy(policy.PARAMETERS);p[k]=v
            with self.assertRaises(Stop):policy.verify(p)
    def test_bound_metadata_requires_explicit_wall_amendment(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=family.Family(Path(tmp))
            try:
                p,a,r=family.bound_documents(f);p['caps']['wall_seconds']=None
                with self.assertRaises(Stop):policy.settings(p,a)
                p['wall_time_profile']=dict(path='mock-wall.json',sha256='a'*64)
                a['wall_time_amendment']=dict(approved=True,**{'from':259200,'to':None},profile=p['wall_time_profile'],authority=dict(path='mock-wall-authority.json',sha256='b'*64))
                a['plan_fingerprint']=r['plan_fingerprint']=fingerprint('h4-newhost-plan-v2',p);r['authorization_digest']=fingerprint('h4-newhost-authorization-v2',a)
                permit=lb.authorize(p,a,r,explicit_launch=True);self.assertIsNone(permit.plan['caps']['wall_seconds'])
                a['wall_time_amendment']['approved']=False
                with self.assertRaises(Stop):policy.settings(p,a)
                a['continuation_amendment']['scope']='run10_encoding_only_resume'
                with self.assertRaises(Stop):policy.structural(p,a)
            finally:f.close()
    def test_runtime_authority_bytes_are_required_and_cap_accounting_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=family.Family(Path(tmp))
            try:
                p,a,r=family.bound_documents(f);p['caps']['wall_seconds']=None;p['wall_time_profile']=dict(path='mock-wall.json',sha256='a'*64)
                a['wall_time_amendment']=dict(approved=True,**{'from':259200,'to':None},profile=p['wall_time_profile'],authority=dict(path='mock-authority.json',sha256='b'*64))
                auth=dict(schema_version='h4-user-unlimited-wall-authority-v1',wall_limit_removal_approved=True,accepted_profile=policy.PARAMETERS,
                          preserve_cumulative_accounting=True,other_limits_unchanged=True,instruction='mock explicit approval')
                def ref(root,entry):return policy.PARAMETERS if entry==p['wall_time_profile'] else auth
                policy.verify_runtime(ROOT,p,a,ref)
                auth['wall_limit_removal_approved']=False
                with self.assertRaises(Stop):policy.verify_runtime(ROOT,p,a,ref)
            finally:f.close()
    def test_second_generation_runtime_gate_with_new_closed_source_set(self):
        # Only these two affected metadata cases, not the completed24 campaign.
        case=family.Resume02Tests('test_runtime_binding_source_budget_and_both_ancestors_pass_mock_only')
        case.setUp()
        try:case.test_runtime_binding_source_budget_and_both_ancestors_pass_mock_only()
        finally:case.tearDown();case.doCleanups()
