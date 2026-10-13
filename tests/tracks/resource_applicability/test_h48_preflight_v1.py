"""Scalar/mock preflight tests. No inputs, science sampling or transpile."""
import itertools
import json
import unittest
from unittest.mock import patch
from copy import deepcopy
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability import h48_preflight_contract_v1 as contract
from trottertracks.resource_applicability.h48_preflight_contract_v1 import (
    h4_plan,h4_cells,h8_input_plan,registrations,resources,fixed_seed)

class PlanTests(unittest.TestCase):
    def test_frozen_H4_panel(self):
        cells=h4_cells();self.assertEqual(len(cells),16)
        self.assertEqual(len({c['id'] for c in cells}),16)
        self.assertEqual({c['prefix'] for c in cells if c['method']=='B2'},{3})
        self.assertEqual({c['prefix'] for c in cells if c['method']=='B1'},{12})
        self.assertEqual({c['formula'] for c in cells if c['method']=='B1'},{'2nd','4th'})
    def test_paired_budget_independent(self):
        p=h4_plan();cs=p['cells'];count=sum(32 if c['R'] else 1 for c in cs)
        self.assertEqual(count,264)
        self.assertEqual(p['caps_proposed']['compile'],2*count+4)
        self.assertEqual(p['caps_proposed']['trajectory'],256)
        self.assertEqual(p['caps_proposed']['occurrence'],sum(c['q']*32 for c in cs if c['R']))
    def test_old_H4_input_policy_preserved(self):
        self.assertEqual(h4_plan()['input_policy'],'EXISTING_H4_FIXED_BYTES_NO_REGENERATION')
        self.assertTrue(h4_plan()['df_policy']['not_asserted_equal_to_tol_only_policy'])
    def test_threads_and_H8_scope(self):
        p=h8_input_plan();self.assertEqual(p['target']['sector_dimension'],4900)
        self.assertEqual(len(p['target']['geometry']),8)
        self.assertEqual(p['target']['nelec_alpha'],4)
        self.assertFalse(p['H8_full_campaign_authorized'])
        self.assertEqual([p['caps_proposed'][k] for k in ('trajectory','occurrence','compile')],[0,0,0])
        self.assertEqual(p['state_policy']['num_threads'],1)
    def test_scalar_kernel_budget(self):
        p=h4_plan();self.assertEqual(p['caps_proposed']['primitive'],3*len(p['coverage']['physical_times']))
        self.assertEqual(p['matrix_free']['num_threads'],1)
        self.assertEqual(p['caps_proposed']['reference_matvec'],36)
    def test_registrations_are_fixed_and_paired(self):
        m={'plan':h4_plan(),'source_commit':'1'*40}
        signals=[{'cell':c,'precision_rows':{'0.05':{'1.0':{'cost_acquisition_eligible':True}}},
                  'signals':{'corrected':{'real':1.,'imag':0.}}} for c in m['plan']['cells']]
        r=registrations(m,signals);self.assertEqual(len(r),264)
        self.assertEqual(len({x['task']['seed'] for x in r}),264)
        self.assertEqual(sum(x['task']['ordinary'] for x in r),2)
        self.assertEqual(sum(x['task']['validate'] for x in r),16)
        self.assertEqual(r,registrations(m,list(reversed(signals))))
        caps=[contract.task_caps(x['task'],m['plan']['caps_proposed']) for x in r]
        for key in ('compile','trajectory','occurrence','control_probe'):
            self.assertEqual(sum(x[key] for x in caps),m['plan']['caps_proposed'][key])
        self.assertEqual(h4_plan('H4_SIGNAL')['caps_proposed']['compile'],0)
    def test_ineligible_not_zero_cost(self):
        m={'plan':h4_plan(),'source_commit':'1'*40}
        signals=[{'cell':c,'precision_rows':{'0.05':{'1.0':{'cost_acquisition_eligible':False}}},
                  'signals':{'corrected':{}}} for c in m['plan']['cells']]
        self.assertEqual(registrations(m,signals),[])
    def test_seed_binds_source(self):
        self.assertNotEqual(fixed_seed('1'*40,'c',0),fixed_seed('2'*40,'c',0))
        self.assertNotEqual(fixed_seed('1'*40,'c',0),fixed_seed('1'*40,'c',1))
    def test_resource_slots(self):
        r=resources();self.assertEqual(len(r['cost_cpu_sets']),8)
        self.assertTrue(set(r['coordinator_cpu_set']).isdisjoint({x[0] for x in r['cost_cpu_sets']}))
        self.assertLessEqual(8*r['worker_rss_stop_bytes']+r['coordinator_rss_reserve_bytes']+r['aggregate_margin_bytes'],r['aggregate_rss_bytes'])
        self.assertEqual(resources(1)['cost_cpu_sets'],[[20]])
        with self.assertRaises(ValueError):resources(9)
    def test_policies_can_be_serialized_without_nonfinite(self):
        self.assertTrue(json.dumps(h4_plan(),allow_nan=False))
        self.assertTrue(json.dumps(h8_input_plan(),allow_nan=False))

class BindingTests(unittest.TestCase):
    def fixture(self):
        raw=b'fixture';import hashlib
        h=hashlib.sha256(raw).hexdigest()
        m={'scope':'H8_INPUT','source_commit':'a'*40,'source_hashes':{'fixture.py':h},
           'profile':{'fixed':'fixture'},'plan':h8_input_plan(),'proposed_resources':resources(1),
           'observation_policy':contract.observation_policy(),
           'base_manifest':{'plan':{'caps_proposed':contract.aggregate_caps()}},
           'registrations':[{'index':0,'task':{'kind':'H8_INPUT'}}],
           'output':'/home/AbeHiromu/projects/h6-handoff-evidence/fixture-unexecuted',
           'H8_full_campaign_authorized':False}
        g={'approved_by_user':True,'manifest_sha256':digest(m),'output':m['output'],
           'H8_full_campaign_authorized':False,
           'authorization_basis':['準備ができたら開始してよい','開始対象は1と3（H8小規模事前検証・H4比較補完）']}
        return raw,h,m,g
    def verify(self,m,g,raw,h):
        with patch.object(contract,'profile',return_value={'fixed':'fixture'}),\
             patch.object(contract,'source_paths',return_value=['fixture.py']),\
             patch.object(contract,'file_hash',return_value=h),\
             patch.object(contract.subprocess,'check_output',return_value=raw):
            return contract.verify_binding('/home/AbeHiromu/projects',m,g)
    def test_valid_metadata_binding(self):
        raw,h,m,g=self.fixture();self.assertTrue(self.verify(m,g,raw,h))
    def test_no_old_or_unapproved_grant(self):
        raw,h,m,g=self.fixture();g['approved_by_user']=False
        with self.assertRaisesRegex(ValueError,'NEW_SCOPE_GRANT_REQUIRED'):self.verify(m,g,raw,h)
    def test_no_full_H8_grant(self):
        raw,h,m,g=self.fixture();g['H8_full_campaign_authorized']=True
        with self.assertRaises(ValueError):self.verify(m,g,raw,h)
    def test_changed_scientific_plan(self):
        raw,h,m,g=self.fixture();m['plan']['target']['geometry_angstrom']=1.2;g['manifest_sha256']=digest(m)
        with self.assertRaisesRegex(ValueError,'PLAN_CHANGED'):self.verify(m,g,raw,h)
    def test_changed_cpu_resources(self):
        raw,h,m,g=self.fixture();m['proposed_resources']['cost_cpu_sets']=[[2]];g['manifest_sha256']=digest(m)
        with self.assertRaisesRegex(ValueError,'RESOURCE_BINDING'):self.verify(m,g,raw,h)
    def test_no_path_escape(self):
        raw,h,m,g=self.fixture();m['output']='/home/AbeHiromu/../fixture';g['output']=m['output'];g['manifest_sha256']=digest(m)
        with self.assertRaisesRegex(ValueError,'HOME_OUTPUT_REQUIRED'):self.verify(m,g,raw,h)

ADAPTER_MODEL='old-model-fixture'
def fixture_callback():return ADAPTER_MODEL

class AdapterTests(unittest.TestCase):
    def test_existing_resource_trace_interface(self):
        import importlib.util,time
        from pathlib import Path
        root=Path(__file__).resolve().parents[3]
        spec=importlib.util.spec_from_file_location('h48_runner_fixture',root/'scripts/resource_applicability/run_track_a_h48_preflight_v1.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        records=[]
        class Writer:
            def write(self,name,row):records.append((name,row))
        trace=module.make_resource_trace(Writer(),time.monotonic())
        with trace.bind():trace.event('fixture')
        self.assertEqual(len(records),1);self.assertEqual(records[0][1]['kind'],'fixture')
        self.assertGreaterEqual(records[0][1]['elapsed_seconds'],0.)
    def test_frozen_module_globals_are_not_mutated(self):
        from trottertracks.resource_applicability.h4_common_panel_port_v1 import bound_function
        f=bound_function(fixture_callback,{'ADAPTER_MODEL':'new-model-fixture'})
        self.assertEqual(f(),'new-model-fixture');self.assertEqual(fixture_callback(),'old-model-fixture')
        with self.assertRaises(ValueError):bound_function(fixture_callback,{'missing_boundary':1})
    def test_H8_solver_mock_budget_before_action(self):
        import numpy as np
        from trottertracks.resource_applicability.ax2b_limits import CallBudget
        b=CallBudget(solver_matvec=1);x=np.zeros(4900,dtype=complex);x[0]=1
        def action(v):
            self.assertEqual(b.used['solver_matvec'],1);return v
        def solver(op,**kwargs):
            self.assertEqual(kwargs['ncv'],40);self.assertEqual(kwargs['which'],'SA')
            np.testing.assert_array_equal(op@x,x);return np.array([1.]),x[:,None]
        result,op,c=contract.bounded_h8_solver(action,4900,x,b,solver=solver)
        self.assertEqual(c.used['calls'],1)
        with self.assertRaises(RuntimeError):op@x
        with self.assertRaises(ValueError):contract.bounded_h8_solver(action,400,x,b,solver=solver)

if __name__=='__main__':unittest.main()
