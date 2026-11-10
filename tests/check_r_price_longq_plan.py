"""Metadata/saved-fixture tests; no scientific evaluation or price arithmetic."""
from copy import deepcopy
from pathlib import Path
import ast
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import plan_r_price_longq as m


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=m.prepare(ROOT)

    def test_fixed_sources_and_A_P1_status_identity(self):
        p=self.plan
        self.assertEqual(p['B_source'],'724d313b9c8445763f0fe9c41feb573df28ba5d5')
        self.assertEqual(len(p['A_P1_coverage']),12)
        self.assertTrue(all(r['N_status']=='PUBLISHED_CONDITIONAL_ROUND_PASS' for r in p['A_P1_coverage']))
        self.assertTrue(all(not r['N_values_recomputed'] for r in p['A_P1_coverage']))
        self.assertTrue(all(type(r['N_X']) is int and type(r['N_Y']) is int for r in p['A_P1_coverage']))
        self.assertEqual(sum(r['C1_cell'] is not None for r in p['A_P1_coverage']),2)
        self.assertTrue(all(not r['C1_independent_validation'] and not r['model_qualified_for_candidate'] for r in p['A_P1_coverage']))

    def test_q_R_joint_roles_and_separate_finite_budgets(self):
        packages={p['id']:p for p in self.plan['packages']}
        self.assertEqual({k:p['new_full_wrapper_compile_budget'] for k,p in packages.items()},
                         dict(Q=16,R=16,QR=24,SHORT=16))
        self.assertTrue(all(not p['automatic_follow_on'] and not p['calibration_fit'] for p in packages.values()))
        self.assertEqual({p['paths_per_cohort'] for p in packages.values()},{4,6})
        primary=self.plan['decision_order'][0]
        self.assertEqual(primary['id'],'P1_DIRECT')
        self.assertEqual(primary['new_full_wrapper_compile_budget'],40)
        self.assertEqual(primary['components'],['Q','QR'])
        for r in self.plan['rows']:
            self.assertEqual(r['role'],'INDEPENDENT_FROZEN_C1_TRANSFER_VALIDATION')
            self.assertFalse(r['calibration'])

    def test_canonical_binding_allocation_and_q_RPE_separation(self):
        rows=self.plan['rows']
        self.assertEqual(len(rows),8)
        self.assertEqual(len({r['price_scope']['join_key'] for r in rows}),8)
        for r in rows:
            c=r['request']['identity']['candidate']
            self.assertEqual(r['candidate_id'],m.p.candidate_id(c))
            self.assertEqual(r['structure'],m.p.structure(c))
            self.assertEqual(r['structure']['event_count'],r['q_PF']*r['R_outer'])
            self.assertEqual(r['T'],r['q_PF']*r['delta'])
            if r['system']=='H6':
                self.assertEqual(len(r['structure']['allocation']),11)
                self.assertEqual(r['q_RPE'],16*r['q_PF'])
                self.assertLess(min(r['structure']['signed_weights']),0)

    def test_no_price_or_uncertainty_fabricated(self):
        for r in self.plan['rows']:
            for k in ['compiled_rz','structural_rz','prediction_rz','residual','all_law_expected_price','R_q_systematic','joint_covariance','total_CI','G']:
                self.assertIsNone(r[k])
            self.assertFalse(r['qualified_for_candidate'])
            self.assertEqual(r['proposed_price_unit'],'FINITE_COHORT_ONLY')
            self.assertTrue(r['paired_XY'])
        bad=deepcopy(self.plan);bad['rows'][0]['compiled_rz']=0
        with self.assertRaisesRegex(ValueError,'scientific value'):m.validate(bad)

    def test_outside_domain_and_unselected_A_candidates_remain_missing(self):
        omitted=[r for r in self.plan['A_P1_coverage'] if r['proposed_package'] is None and r['C1_cell'] is None]
        self.assertEqual({(r['system'],r['m'],r['R_outer']) for r in omitted},
            {('H4',0,2),('H4',0,8),('H4',6,256),('H6',8,256)})
        self.assertTrue(all(r['C'] is None and r['G'] is None for r in omitted))
        self.assertEqual([r['structure']['event_count'] for r in self.plan['deferred_final_rounds']],
                         [33554432,524288])
        self.assertTrue(all(r['proposed_direct_compiles']==0 for r in self.plan['deferred_final_rounds']))

    def test_tampered_R_or_allocation_is_rejected(self):
        for kind in ['R','allocation']:
            bad=deepcopy(self.plan)
            if kind=='R':bad['rows'][0]['request']['identity']['candidate']['R_outer']*=2
            else:bad['rows'][0]['structure']['allocation'][0]+=1
            with self.assertRaisesRegex(ValueError,'binding'):m.validate(bad)

    def test_old_validation_roles_not_relabelled_and_no_rare_package(self):
        self.assertFalse(self.plan['old_validation_relabelled_as_calibration'])
        self.assertEqual(sum(c['role']=='VALIDATION' for c in self.plan['existing_C1_cells']),2)
        self.assertEqual(self.plan['rare_dedicated_experiments'],0)
        bad=deepcopy(self.plan);bad['rare_dedicated_experiments']=1
        with self.assertRaisesRegex(ValueError,'outside'):m.validate(bad)

    def test_metadata_entry_has_no_science_or_execution_call(self):
        tree=ast.parse(Path(m.__file__).read_text())
        calls={getattr(n.func,'attr',getattr(n.func,'id','')) for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(calls & {'sample_requests','price_requests','fit_model','evaluate_cell','transpile','evaluate_menu','run_evaluation','predict_path'})
        self.assertEqual(self.plan['new_price_records'],[])
        self.assertEqual(self.plan['scientific_work_this_task'],0)


if __name__=='__main__':unittest.main(verbosity=2)
