"""H6 D1 delta tests on artificial coefficients/signals, never saved H6 data."""
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import tempfile
import math
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import h6_df19_rpe_d1 as d1


def point(later=False,method='PR'):
    from check_h6_df19_rpe_connection import point as toy
    p=toy(later)
    if method=='FULL':
        identity='a4f5ffc0f320644dadfa' if later else '76601611270ed9f0677d'
        p['candidate'].update(id=identity,method='FULL',prefix=19,R_outer=None,K=None)
    logB=math.log(2.) if method=='PR' else 0.
    A=math.exp(-logB)
    p['saved_signal']=dict(reference=dict(real=1.,imag=0.),
        corrected=dict(real=.999,imag=.0001),raw=dict(real=A*.999,imag=A*.0001),
        total_signed_bias=dict(real=-.001,imag=.0001),log_B=logB)
    p['saved_allowance']=dict(u=1e-6,log_B_margin=1e-8)
    endpoint=dict(candidate=deepcopy(p['candidate']),axes={a:dict(allowance=2e-6)
                  for a in ('real','imag')})
    if method=='FULL':p['prices']=p['prices'][:1]
    for price in p['prices']:
        price['candidate']=deepcopy(p['candidate'])
        price['cohort']='DETERMINISTIC' if method=='FULL' else (
            'EXPLORATION' if price['n']==9 else 'CONFIRMATION')
        price['classification']='DIRECT_COMPILED_DETERMINISTIC' if method=='FULL' else 'DIRECT_COMPILED_FINITE_SAMPLE'
    return d1.hydrate_saved_allowances(p,endpoint)


class H6D1Tests(unittest.TestCase):
    def test_prior_is_exact_same_function_and_needs_no_state_energy(self):
        self.assertIs(d1.coefficient_prior,d1.scalar.coefficient_prior)
        p=d1.coefficient_prior(0.,[[0.,0.],[0.,0.]],[-1.],[[[1.,0.],[0.,2.]]],[0])
        self.assertLessEqual(p['energy_prior_ha'][0],-4.)
        self.assertFalse(p['uses_state_energy_or_E0'])

    def test_only_two_h_values_and_stop_instead_of_smaller_h(self):
        c=d1.configuration([-100.,100.],energy_ha=.1)
        self.assertEqual(c['status'],'STOP_NO_ALLOWED_BASE_TIME')
        self.assertEqual([x['base_time'] for x in c['prior_attempts']],[.8,.1])
        self.assertIsNone(c['horizon'])
        c=d1.configuration([-10.,10.],energy_ha=.1)
        self.assertEqual(c['base_time'],.1)

    def test_h6_coefficient_loader_never_decodes_state_vector(self):
        import numpy as np
        indices=sorted(sum(1<<i for i in a+b) for a in combinations(range(0,12,2),3)
                       for b in combinations(range(1,12,2),3))
        with tempfile.TemporaryDirectory(prefix='h6_d1_toy_') as temp:
            path=Path(temp)/'toy.npz'
            np.savez(path,constant=0.,one_body=np.zeros((12,12)),lambdas=np.zeros(19),
                g_matrices=np.zeros((19,12,12)),sector_basis_indices=indices,
                state_vector=np.array([object()],dtype=object))
            c,h,w,g,occ=d1.load_coefficient_payload(path)
        self.assertEqual((len(h),len(w),len(g),len(occ)),(12,19,19,6))
        self.assertEqual(sum(i%2==0 for i in occ),3)

    def test_two_times_and_unknown_actual_lane_reuse_same_scalar_glue(self):
        config=d1.configuration([-2.,1.],energy_ha=.1)
        for later in (False,True):
            r=d1.analyze_point(point(later),config,state_energy=0.,state_residual=1e-8)
            self.assertEqual(r['m'],int(later));self.assertEqual(r['q_PF'],1)
            self.assertEqual(r['ideal_numerical_pilot']['status'],'CONDITIONAL_ROUND_PASS')
            self.assertIsNone(r['actual_implementation']['N_X'])
            self.assertFalse(r['reference_diagnostic']['used_for_prior_or_branch_choice'])
            self.assertGreaterEqual(r['error_disk']['preserved_u_X'],2e-6)

    def test_partial_costs_separate_pf_and_both_pr_cohorts_and_missing_rounds(self):
        config=d1.configuration([-10.,10.],energy_ha=.1)
        points=[point(later,method) for later in (False,True) for method in ('FULL','PR')]
        rows=[d1.analyze_point(p,config,state_energy=0.,state_residual=1e-8) for p in points]
        out=d1.partial_resources(points,rows,config)
        self.assertEqual([(x['method'],x['cohort']) for x in out],
                         [('PF','DETERMINISTIC'),('PR','EXPLORATION'),('PR','CONFIRMATION')])
        for r in out:
            self.assertEqual(r['included_rounds'],(3,4))
            self.assertTrue({0,1,2}.issubset(r['missing_rounds']))
            self.assertTrue(r['partial_only']);self.assertFalse(r['full_rpe_energy_success'])
            self.assertIsNone(r['actual_totals']);self.assertIsNone(r['total_confidence_interval'])


if __name__=='__main__':unittest.main()
