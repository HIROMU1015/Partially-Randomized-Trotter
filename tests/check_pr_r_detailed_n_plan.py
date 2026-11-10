"""Static public fixtures and artificial operation counts only, no science."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import plan_pr_r_detailed_n as m


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=m.prepare(ROOT)

    def test_no_molecular_numeric_imports_or_execution_option(self):
        code="import plan_pr_r_detailed_n as m;m.prepare();import sys;assert not any(s in sys.modules for s in ('numpy','scipy','qiskit'))"
        env=__import__('os').environ.copy();env['PYTHONPATH']='src:scripts'
        subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,check=True)
        with patch.object(m.screening,'screen_normalization',side_effect=AssertionError('No normalization')):
            self.assertEqual(m.prepare()['science_calls_this_task'],0)
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/plan_pr_r_detailed_n.py'),'--execute'],capture_output=True)
        self.assertNotEqual(result.returncode,0)

    def test_full24_18_proposed_6_deferred_no_scientific_adoption(self):
        rows=self.plan['rows'];self.assertEqual(len(rows),24)
        self.assertEqual(sum(r['selected_for_proposal'] for r in rows),18)
        self.assertEqual(len({r['candidate_id'] for r in rows}),24)
        self.assertIsNone(self.plan['R_upper_bound']);self.assertFalse(self.plan['range_complete'])
        for r in rows:
            self.assertFalse(r['execution_authorized']);self.assertIsNone(r['N']);self.assertIsNone(r['scientific_PASS_FAIL'])
            self.assertEqual(r['allowance_policy'],m.POLICY)

    def test_a0_values_verbatim_and_law_declarations_distinct(self):
        raw=m.p.SavedArtifacts(ROOT).read(m.SOURCES['A0'])['records']
        lookup={r['candidate_id']:r for r in raw}
        for r in self.plan['rows']:
            self.assertEqual(r['A0_record'],lookup[r['candidate_id']])
            self.assertEqual(r['N_request']['identity'],r['A0_record']['request']['identity'])
            self.assertNotEqual(r['N_request']['law_binding'],r['A0_record']['request']['law_binding'])
        self.assertFalse(self.plan['old_N_comparable_to_new_policy'])

    def test_h6_m4_missing_pair_and_norm_are_not_zero_filled(self):
        baseline=next(r for r in self.plan['rows'] if r['system']=='H6' and r['m']==4 and r['baseline'])
        self.assertFalse(baseline['reference']['saved_pair_shape_present'])
        self.assertFalse(baseline['signal']['baseline_signal_evidence_reusable'])
        self.assertTrue(baseline['signal']['proposed_new_finite_mean'])
        self.assertEqual(baseline['reference']['proposed_new_actions'],2)
        for r in self.plan['rows']:
            if (r['system'],r['m'])!=('H6',4):
                self.assertTrue(r['reference']['saved_pair_shape_present'])
                self.assertEqual(r['reference']['proposed_new_actions'],0)

    def test_finite_call_counts_and_baselines_first(self):
        a,b,c=self.plan['packages']
        self.assertEqual(a['proposed_calls'],dict(new_finite_means=9,internal_exact_tail_controls=9,
            saved_baseline_scalar_only=3,scalar_phase_radius_N=12,axis_N=24,new_reference_actions=2,
            sampling=0,compile=0,C=0,G=0))
        for pkg in (b,c):
            self.assertEqual(pkg['proposed_calls']['new_finite_means'],2)
            self.assertEqual(pkg['proposed_calls']['saved_baseline_scalar_only'],1)
            self.assertEqual(pkg['proposed_calls']['new_reference_actions'],0)
        for pkg in (a,b,c):
            lookup={r['candidate_id']:r for r in self.plan['rows']};order=[lookup[c]['baseline'] for c in pkg['execution_order']]
            self.assertEqual(order,sorted(order,reverse=True));self.assertFalse(pkg['automatic_follow_on'])

    def test_operation_counts_use_source_loops_and_no_event_arrays(self):
        row=dict(system='H4',q_PF=3,R_outer=5,prefix=1,
            structure=dict(signed_weights=[1],allocation=[5],event_count=15))
        v=m.operations(row)
        self.assertEqual((v['finite_two_column_updates'],v['raw_vector_divisions'],v['head_three_column_actions'],v['intermediate_checks']),(15,15,12,28))
        self.assertEqual(v['finite_dense_dot_terms'],2*15*36**2)
        self.assertFalse(v['event_sized_arrays_required']);self.assertIsNone(v['wall_seconds_prediction'])
        bad=deepcopy(row);bad['structure']['allocation']=[4]
        with self.assertRaises(ValueError):m.operations(bad)

    def test_signed_allocation_and_C_exact_domain(self):
        eligible=[]
        for r in self.plan['rows']:
            self.assertEqual(sum(r['structure']['allocation']),r['R_outer'])
            if r['system']=='H6':
                self.assertGreaterEqual(r['m'],4);self.assertEqual(len(r['structure']['allocation']),11)
            if r['C_coverage']['phase1_cell']:eligible.append(r['candidate_id'])
            self.assertFalse(r['C_coverage']['qualified_for_candidate'])
        self.assertEqual(set(eligible),{'2d2af6ca830dff14a695','4269eafc8c0ef8922c23'})

    def test_corrected_integer_inventory_is_copy_only(self):
        saved=m.p.SavedArtifacts(ROOT).read(m.SOURCES['H4_N'])['shots']
        for r in self.plan['H4_corrected_N_inventory']:
            self.assertIn(r,saved)
            self.assertIs(type(r['plan']['N_X']),int)
        final=next(r for r in self.plan['H4_corrected_N_inventory'] if r['identity']['m']==12)
        self.assertGreater(final['plan']['N_X'],2**53)
        restored=json.loads(json.dumps(self.plan));self.assertEqual(restored['H4_corrected_N_inventory'],self.plan['H4_corrected_N_inventory'])


if __name__=='__main__':unittest.main(verbosity=2)
