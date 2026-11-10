"""New D1 glue tests on artificial coefficient/scalar fixtures only."""
from pathlib import Path
import math
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import h4_df10_rpe_d1 as d1


class Tests(unittest.TestCase):
    def test_exact_coefficient_ground_prior(self):
        p=d1.coefficient_prior(0.,[[-1,0],[0,2]],[.5],[[[1,0],[0,2]]],[0])
        self.assertLessEqual(p['energy_prior_ha'][0],-.5)
        self.assertGreaterEqual(p['energy_prior_ha'][1],-.5)
        self.assertEqual(p['determinant_variational_upper_ha'],-.5)
        self.assertFalse(p['uses_state_energy_or_E0'])

    def test_complex_determinant_cross_term_and_signed_square(self):
        p=d1.coefficient_prior(0.,[[-1,0],[0,2]],[.5],[[[1,1j],[-1j,2]]],[0])
        self.assertEqual(p['determinant_variational_upper_ha'],0.)
        n=d1.coefficient_prior(0.,[[0,0],[0,0]],[-1.],[[[1,0],[0,2]]],[0])
        self.assertLessEqual(n['energy_prior_ha'][0],-4.)
        self.assertGreaterEqual(n['energy_prior_ha'][1],-4.)

    def test_nonhermitian_or_invalid_sector_rejected(self):
        for h,occ in [([[0,1],[0,0]],[0]),([[0,0],[0,0]],[0,0])]:
            with self.assertRaises(ValueError):d1.coefficient_prior(0,h,[],[],occ)

    def test_preferred_h_and_uniform_full_horizon_alpha(self):
        c=d1.configuration([-2.,1.])
        self.assertEqual(c['base_time'],.8)
        self.assertEqual(c['D1_m'],0)
        self.assertLessEqual(c['alpha_sum_all_rounds'],.05)
        self.assertLessEqual(c['horizon']['achieved_energy_resolution_ha'],d1.CHEMICAL_ACCURACY_HA)
        self.assertTrue(c['horizon']['previous_power_fails'])

    def test_conservative_prior_rejects_large_h(self):
        c=d1.configuration([-10.,10.])
        self.assertEqual(c['base_time'],.1)
        self.assertEqual(c['D1_m'],3)
        self.assertFalse(c['prior_attempts'][0]['qualified'])
        self.assertNotEqual(c['alpha_axis'],.025)
        self.assertLessEqual(c['alpha_sum_all_rounds'],.05)

    def test_saved_scalar_glue_ideal_and_unknown_actual_errors(self):
        from copy import deepcopy
        from check_h4_df10_rpe_connection import point
        p=point();p['candidate'].update(T=.8,id=d1.IDS[1])
        p['prices'][0]['candidate']=deepcopy(p['candidate'])
        p['prices'][0]['cohort']='ARTIFICIAL'
        p['prices'][0]['mean_metrics']={a:dict(rz_count=10.) for a in ('cosine','sine')}
        p['saved_signal']=dict(reference=dict(real=1.,imag=0.),
            corrected=dict(real=.999,imag=.0001),raw=dict(real=.4995,imag=.00005),
            total_signed_bias=dict(real=-.001,imag=.0001),log_B=math.log(2.))
        p['saved_allowance'].update(u=1e-6,log_B_margin=1e-8)
        p['saved_finite_time_endpoint']=dict(axes={a:dict(allowance=1e-6) for a in ('real','imag')})
        r=d1.analyze_point(p,d1.configuration([-2.,1.]),state_energy=0.,state_residual=0.)
        self.assertEqual(r['ideal_numerical_pilot']['status'],'CONDITIONAL_ROUND_PASS')
        self.assertIsNone(r['actual_implementation']['N_X'])
        self.assertIsNone(r['error_disk']['unconditional_ground_identification_error'])
        self.assertFalse(r['round_resource_product_computed'])
        self.assertFalse(r['reference_diagnostic']['used_for_prior_or_branch_choice'])
        self.assertGreater(r['normalization']['raw_margin'],0.)

    def test_tiny_h_alternative_and_no_solver_energy_prior_input(self):
        c=d1.configuration([-100.,100.])
        self.assertLess(c['base_time'],.1)
        self.assertEqual(math.ldexp(c['base_time'],c['D1_m']),.8)
        with self.assertRaises(TypeError):d1.coefficient_prior(0,[[0]],[],[],[0],state_energy=-999)


if __name__=='__main__':unittest.main()
