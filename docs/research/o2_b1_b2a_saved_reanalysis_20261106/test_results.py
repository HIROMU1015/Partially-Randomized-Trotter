"""Cross-check saved-value outputs against frozen identities and closed-form sums."""
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import unittest

P = Path(__file__).resolve().parent


class ResultChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.norm = list(csv.DictReader((P/'B2A_NORMALIZATION_RESULTS.csv').read_text().splitlines()))
        cls.cx = list(csv.DictReader((P/'B1_CX_SENSITIVITY.csv').read_text().splitlines()))
        cls.fixed = json.loads((P/'B1_FIXED_COMPARISONS.json').read_text())['comparisons']

    def test_full_frozen_normalization_domain_retained(self):
        fixed = list(csv.DictReader((P/'B2A_FIXED_CANDIDATES.csv').read_text().splitlines()))
        self.assertEqual([(r['family'],r['candidate_id']) for r in self.norm],[(r['family'],r['candidate_id']) for r in fixed])
        self.assertTrue(all(r['exact_status']=='EXACT_REPRODUCED' for r in self.norm))

    def test_independent_factorial_finite_sum(self):
        # Direct powers/factorials differ from the original log/exp/lgamma path.
        for r in self.norm:
            w,counts = json.loads(r['stage_weights']),json.loads(r['integer_allocation'])
            lam,T,q,K = float(r['lambda_R']),float(r['T']),int(r['q']),int(r['K'])
            terms=[]
            for weight,count in zip(w,counts):
                tau=lam*((T*weight/q)/count)
                beta=math.fsum(abs(tau)**k/math.factorial(k)*math.hypot(1,tau/(k+1)) for k in range(0,K+1,2))
                terms.append(q*count*math.log(beta))
            self.assertLessEqual(abs(math.fsum(terms)-float(r['saved_log_B'])),1e-10)

    def test_all_fixed_candidate_cohort_N_bindings_retained(self):
        actual={r['comparison_id']:r for r in self.cx}
        for c in self.fixed:
            row=actual[c['comparison_id']]
            for side,a in zip(('left','right'),c['arms'].values()):
                self.assertEqual(row[side+'_id'],a['candidate_id'])
                self.assertEqual(row[side+'_cohort'],a['cohort'])
                self.assertEqual(int(row[side+'_N_cos']),int(a['N_cos']))
                self.assertEqual(int(row[side+'_N_sin']),int(a['N_sin']))

    def test_missing_raw_does_not_gain_zero_SE(self):
        for r in self.cx:
            if r['study'].startswith('DF_'):
                self.assertEqual(r['uncertainty_status'],'UNCERTAINTY_UNRESOLVED')
                self.assertEqual(r['CX_difference_SE'],'')

    def test_no_price_pooling_or_axis_loss_in_paired_domains(self):
        for r in self.cx:
            if r['study']=='PAULI_LATEST_15':
                expected={'H2':(96,384),'H4':(48,384),'H6':(64,48)}[r['system']]
                self.assertEqual((int(r['left_n']),int(r['right_n'])),expected)
                self.assertGreaterEqual(float(r['CX_difference_SE']),0)
            if r['study']=='H8_STAGE_C':
                self.assertEqual((int(r['left_n']),int(r['right_n'])),(9,1))
                self.assertEqual(r['right_CX_SE'],'')

    def test_RZ_reference_checks_and_no_silent_failures(self):
        e=json.loads((P/'EXECUTION.json').read_text())
        self.assertEqual(len(e['RZ_checks']),2*len(self.fixed))
        self.assertEqual(e['stops'],[])
        self.assertTrue(all(r['RZ_check'] in ('EXACT_EQUAL','FLOAT_PRECISION_MATCH') for r in e['RZ_checks']))
        self.assertEqual([r['study'] for r in self.cx if r['direction_status']=='DATA_UNAVAILABLE'],['H4_GENERIC_DF_PAULI_CROSS'])


if __name__=='__main__':
    unittest.main()
