"""B6b artificial theorem/contract/accounting tests; no H4 totals or compile."""
import ast
import copy
import importlib.util
import inspect
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_track_d_h4_reference as boot
boot.bootstrap(os.environ['TRACK_D_ORIGIN'])
from trottertracks import track_d_b6b_bc as bc
from trottertracks import track_d_b6b_toy as toy


def premises():
    return {k:True for k in ('same_V','actual_adjoint','vacuum_eigenray_with_chi','same_initial_system_state',
        'same_whole_path_law','same_XY_convention','fresh_independent_axis_trials','same_target_decoder','law_is_B4_ideal_product')}


def synthetic():
    handoff=dict(binding=dict(original_h=bc.PACKET,cdf=bc.CDF),rows=[dict(m=m,q=2**m,N_X=3+m%3,N_Y=3+m%3) for m in range(13)])
    prices=[dict(m=m,q=2**m,axis=a,method=method,binding=bc.binding(),scope=bc.SCOPE,compiler=bc.P0,
        expectation_basis='qualified_expectation',mean={k:float(m+5+(method=='reference')) for k in bc.METRICS},
        source='ARTIFICIAL_ACCOUNTING_FIXTURE',estimate_unit='whole_path_one_interrogation',
        dataset='ARTIFICIAL_ACCOUNTING_ONLY') for method in bc.METHODS for m in range(13) for a in ('X','Y')]
    return handoff,prices


class Accounting(unittest.TestCase):
    def test_common_ideal_N_does_not_require_actual_qualification(self):
        p=premises();self.assertTrue(bc.common_N(p)['common_N'])
        self.assertFalse(bc.common_N(p,lane='actual')['common_N'])

    def test_same_finite_bit_law_does_not_imply_B4_transfer(self):
        p=premises();p['law_is_B4_ideal_product']=False
        self.assertEqual(bc.common_N(p)['status'],'SAME_LAW_BUT_B4_N_TRANSFER_UNQUALIFIED')
        p['same_whole_path_law']=False;self.assertFalse(bc.common_N(p)['common_N'])

    def test_synthetic_total_equals_weighted_paired_difference(self):
        handoff,prices=synthetic();r=bc.aggregate_registered(handoff,prices,premises())
        expected=sum(2*x['N_X'] for x in handoff['rows'])
        self.assertEqual(r['paired_delta'],{k:expected for k in bc.METRICS})
        self.assertIsNone(r['paired_total_covariance_diagonal'])
        self.assertFalse(r['covariance_missing_is_zero'])

    def test_shared_round_and_method_covariance_is_retained(self):
        h,p=synthetic();cov={k:np.ones((52,52))*.04 for k in bc.METRICS}
        r=bc.aggregate_registered(h,p,premises(),covariance_by_metric=cov)
        for v in r['paired_total_covariance_diagonal'].values():self.assertAlmostEqual(v,0.,places=10)
        v=np.r_[np.zeros(26),np.ones(26)];cov={k:np.outer(v,v) for k in bc.METRICS}
        r=bc.aggregate_registered(h,p,premises(),covariance_by_metric=cov)
        expected=sum(2*x['N_X'] for x in h['rows'])**2
        self.assertEqual(r['paired_total_covariance_diagonal']['cx_count'],expected)

    def test_missing_unknown_or_finite_price_rejected(self):
        h,p=synthetic()
        with self.assertRaises(ValueError):bc.aggregate_registered(h,p[:-1],premises())
        for key,value in [('mean',None),('expectation_basis','finite_sample')]:
            q=copy.deepcopy(p);q[0][key]=value
            with self.assertRaises(ValueError):bc.aggregate_registered(h,q,premises())

    def test_scope_identity_q_and_compiler_rejected(self):
        h,p=synthetic()
        for key,value in [('scope','kernel_only'),('binding',{}),('compiler',{}),('q',99)]:
            q=copy.deepcopy(p);q[0][key]=value
            with self.assertRaises(ValueError):bc.aggregate_registered(h,q,premises())
        h['binding']['original_h']='other'
        with self.assertRaises(ValueError):bc.aggregate_registered(h,p,premises())

    def test_width_attenuation_and_actual_are_not_hidden_defaults(self):
        h,p=synthetic();q=copy.deepcopy(p);q[0]['mean']['peak_qubits']=17
        with self.assertRaises(ValueError):bc.aggregate_registered(h,q,premises())
        with self.assertRaises(ValueError):bc.aggregate_registered(h,p,premises(),claim='actual')
        r=bc.aggregate_registered(h,p,premises());self.assertFalse(r['attenuation_price_multiplier'])
        self.assertFalse(r['depth_is_hardware_runtime'])

    def test_fixed_B4_equal_axis_rule_not_silently_generalized(self):
        h,p=synthetic();h['rows'][0]['N_Y']+=1
        with self.assertRaises(ValueError):bc.aggregate_registered(h,p,premises())
        h,p=synthetic();h['rows'][0]['N_X']=None
        with self.assertRaises(ValueError):bc.aggregate_registered(h,p,premises())

    def test_legacy_aggregation_AST_matches_B4(self):
        path='src/trottertracks/c1_stage3_accuracy.py'
        old=subprocess.check_output(['git','show',bc.B4+':'+path],cwd=ROOT,text=True)
        extract=lambda s:ast.dump(next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name=='aggregate_totals'),include_attributes=False)
        self.assertEqual(extract(old),extract((ROOT/path).read_text()))
        self.assertEqual(extract(old),extract(inspect.getsource(bc.accounting.aggregate_totals)))


class Theorem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        spec=json.loads((ROOT/'docs/research/track_d_c7_2_cx_structure_20261107/CASES.json').read_text())
        support=boot.load_file('b6b_exterior_support',ROOT/'scripts/native_df_qiskit_validation_support.py')
        cls.cases,cls.initial=toy.build_cases(spec,support.exterior_fock_lift)

    def test_all_inputs_XY_coherence_and_vacuum_conditions(self):
        for case in self.cases:
            wrappers,r=toy.check_case(case);self.assertEqual(len(wrappers),4);self.assertEqual(r['status'],'PASS')

    def test_whole_V_ray_does_not_require_half_ray_for_signal(self):
        _,r=toy.check_case(self.cases[-1])
        self.assertFalse(r['DF_source_eligible']);self.assertGreater(r['reference_nonvacuum_Frobenius'],.01)
        self.assertGreater(r['half_A_vacuum_leakage'],.01)

    def test_invalid_conditions_are_detected(self):
        rows=toy.negative_controls(self.cases[0]);self.assertEqual(len(rows),5)
        self.assertTrue(all(r['detected'] for r in rows))


if __name__=='__main__':unittest.main()
