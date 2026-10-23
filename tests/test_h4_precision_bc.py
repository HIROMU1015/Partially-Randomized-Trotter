"""Artificial B arithmetic and C enclosure prototypes; no real H4 input."""
from copy import deepcopy
from fractions import Fraction as Q
import importlib.util
import math
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'src/trotterlib'/file)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


b=module('b_fixture','h4_precision_budget.py')
c=module('c_fixture','h4_enclosure_prototype.py')


def fixture_rows():
    result=[]
    counts=dict(pauli_jw_PF=1,pauli_jw_PR=3,df_native_PF=3,df_native_PR=4)
    for case in b.CASES:
        for index,epsilon in enumerate(b.EPSILONS):
            for axis in b.AXES:
                e=epsilon/math.sqrt(2);bias=1e-5 if axis=='cosine' else -1e-5
                u=e/20 if index<counts[case] else 2*e
                result.append(dict(case=case,epsilon_sig=epsilon,axis=axis,epsilon_axis=e,
                    signed_bias=bias,absolute_saved_bias=abs(bias),saved_u=u,
                    saved_headroom=e-abs(bias)-u,saved_status='UNDETERMINED',saved_shots=None,
                    saved_N_total=None,saved_G=None,group='R' if index<counts[case] else 'H',
                    normalization_as_saved=dict(B=1.,log_B=0.,normalization_certificate_verified=False)))
    return result


class BudgetArithmetic(unittest.TestCase):
    def test_forty_axes_independent_decimal_checks_and_no_NG(self):
        rows=fixture_rows();b.validate_registry(rows)
        result=[b.diagnose_axis(row) for row in rows]
        self.assertEqual(sum(x['independent_arithmetic']['status']=='PASS' for x in result),40)
        self.assertTrue(all(x['official_N'] is None and x['official_G'] is None and x['official_status']=='UNDETERMINED' for x in result))

    def test_closed_exact_boundary_and_neighbors(self):
        s=Q(3,7);rho=Q(1,100);u=rho*s/(1+rho)
        self.assertEqual(u/(s-u),rho)
        self.assertGreater((u+Q(1,10**20))/(s-u-Q(1,10**20)),rho)
        self.assertLess((u-Q(1,10**20))/(s-u+Q(1,10**20)),rho)

    def test_source_float_equality_and_nextafter_boundary(self):
        e=1.01;u=.01
        self.assertLessEqual(u/(e-u),.01)
        up=math.nextafter(u,math.inf);down=math.nextafter(u,0.)
        self.assertGreater(up/(e-up),.01)
        self.assertLess(down/(e-down),.01)

    def test_binary_rho_and_decimal_rho_are_separately_recorded(self):
        r=b.diagnose_axis(fixture_rows()[0])
        self.assertNotEqual(r['u_max_source_binary_rho']['exact'],r['u_max_ideal_decimal_rho']['exact'])
        self.assertLessEqual(Q(r['u_budget_inward_display']),Q(r['u_max_conservative']['exact']))

    def test_reduction_and_rate_identities(self):
        r=b.diagnose_axis(fixture_rows()[0]);u=Q(r['saved_u']);target=Q(r['u_max_conservative']['exact'])
        self.assertEqual(Q(r['required_u_reduction']['exact']),u-target)
        self.assertEqual(Q(r['required_u_reduction_percent']['exact']),100*(u-target)/u)
        self.assertEqual(Q(r['required_u_improvement_factor']['exact']),u/target)

    def test_positive_headroom_is_not_sufficient(self):
        r=b.diagnose_axis(fixture_rows()[0])
        self.assertGreater(r['saved_headroom'],0)
        self.assertGreater(Q(r['current_guard_ratio_exact']['exact']),Q(1,100))

    def test_H_group_has_positive_budget_if_bias_has_slack(self):
        row=next(x for x in fixture_rows() if x['group']=='H');r=b.diagnose_axis(row)
        self.assertLessEqual(r['saved_headroom'],0)
        self.assertGreater(Q(r['u_max_conservative']['exact']),0)
        self.assertIsNone(r['current_guard_ratio_exact'])

    def test_improving_u_cannot_clear_exhausted_fixed_bias(self):
        row=fixture_rows()[0];row.update(signed_bias=row['epsilon_axis'],absolute_saved_bias=row['epsilon_axis'],saved_u=1.,group='H')
        row['saved_headroom']=row['epsilon_axis']-abs(row['signed_bias'])-row['saved_u']
        r=b.diagnose_axis(row)
        self.assertTrue(r['fixed_point_bias_makes_guard_impossible'])
        self.assertIsNone(r['u_max_conservative'])

    def test_signed_bias_and_saved_subtraction_mismatch_refused(self):
        row=fixture_rows()[0];row['absolute_saved_bias']+=.001
        with self.assertRaisesRegex(ValueError,'SIGNED_ABSOLUTE_BIAS'):b.diagnose_axis(row)
        row=fixture_rows()[0];row['saved_headroom']+=.01
        with self.assertRaisesRegex(ValueError,'SAVED_BINARY_HEADROOM'):b.diagnose_axis(row)

    def test_nonfinite_changed_official_state_and_missing_axes_refused(self):
        row=fixture_rows()[0];row['saved_u']=float('nan')
        with self.assertRaisesRegex(ValueError,'FINITE_SCALAR'):b.diagnose_axis(row)
        row=fixture_rows()[0];row['saved_N_total']=1
        with self.assertRaisesRegex(ValueError,'ORIGINAL_STATE'):b.diagnose_axis(row)
        rows=fixture_rows()
        for bad in [rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]]:
            with self.assertRaisesRegex(ValueError,'FORTY_AXIS'):b.validate_registry(bad)

    def test_summary_preserves_R22_H18(self):
        s=b.summarize([b.diagnose_axis(row) for row in fixture_rows()])
        self.assertEqual((s['groups']['R']['axes'],s['groups']['H']['axes']),(22,18))
        self.assertTrue(s['official_N_G_remain_null'])

    def test_regression_against_unchanged_legacy_accounting_on_artificial_data(self):
        legacy=module('legacy_accounting_regression','fixed_signal_accounting.py')
        target=Q(1,101)/2
        result=legacy.fixed_signal_shots(epsilon_sig=2.,bias={a:0. for a in b.AXES},
            allowance={a:float(target) for a in b.AXES},log_B_upper=0.,
            evidence_kind='EMPIRICAL',evidence_ref='ARTIFICIAL_ONLY',
            epsilon_axes={a:1. for a in b.AXES})
        self.assertTrue(result['usable'])
        self.assertFalse(result['certificate_verified'])
        self.assertTrue(all(x['allowance']/x['headroom']<=.01 for x in result['axes'].values()))


class EnclosureProofPrototypes(unittest.TestCase):
    def test_interval_endpoints_and_zero_divisor(self):
        a=c.Interval(-2,3);b_=c.Interval(-5,-1)
        self.assertEqual(((a*b_).lo,(a*b_).hi),(-15,10))
        with self.assertRaisesRegex(ValueError,'ZERO'):a/c.Interval(-1,1)

    def test_sqrt_outward_by_exact_squared_endpoint_inequalities(self):
        for x in [Q(0),Q(1),Q(2),Q(17,31),Q(123,7)]:
            r=c.sqrt_point(x,bits=64)
            self.assertLessEqual(r.lo*r.lo,x);self.assertGreaterEqual(r.hi*r.hi,x)
        self.assertEqual(c.sqrt_point(Q(9,16)).lo,Q(3,4))

    def test_log_range_reduction_reciprocal_and_known_zero(self):
        self.assertTrue(c.log_point(1).contains(0))
        self.assertTrue((c.log_point(8)-3*c.log_point(2)).contains(0))
        self.assertTrue((c.log_point(Q(1,8))+c.log_point(8)).contains(0))

    def test_finite_law_zero_and_exact_weight_squares(self):
        z=c.finite_law_prototype(0)
        self.assertEqual((z['B'].lo,z['B'].hi),(1,1))
        self.assertTrue(z['log_B'].contains(0))
        tau=Q(1,10);r=c.finite_law_prototype(tau,bits=80)
        for k,w in zip(range(0,5,2),r['weights']):
            target=tau**(2*k)/Q(math.factorial(k)**2)*(1+(tau/Q(k+1))**2)
            self.assertLessEqual(w.lo*w.lo,target);self.assertGreaterEqual(w.hi*w.hi,target)

    def test_matrix_exp_zero_and_nilpotent(self):
        zero=c.exponential_prototype([[0,0],[0,0]])
        for i in range(2):
            for j in range(2):self.assertTrue(zero[i][j].contains(int(i==j)))
        result=c.exponential_prototype([[0,Q(1,4)],[0,0]])
        for a,row in zip(result,[[1,Q(1,4)],[0,1]]):
            for x,y in zip(a,row):self.assertTrue(x.contains(y))

    def test_skew_rotation_encloses_independent_alternating_series(self):
        t=Q(1,4);result=c.exponential_prototype([[0,-t],[t,0]],degree=20)
        sin=sum(((-1)**k*t**(2*k+1)/math.factorial(2*k+1) for k in range(60)),Q(0))
        cos=sum(((-1)**k*t**(2*k)/math.factorial(2*k) for k in range(60)),Q(0))
        sin_interval=c.Interval(sin,sin+t**121/math.factorial(121))
        cos_interval=c.Interval(cos,cos+t**120/math.factorial(120))
        self.assertTrue(result[0][0].contains(cos_interval))
        self.assertTrue(result[1][0].contains(sin_interval))

    def test_noncommuting_factor_order_is_not_reversed(self):
        A=[[0,Q(1,4)],[0,0]];B=[[0,0],[Q(1,3),0]]
        AB=c.multiply(c.exponential_prototype(A),c.exponential_prototype(B))
        BA=c.multiply(c.exponential_prototype(B),c.exponential_prototype(A))
        self.assertTrue(AB[0][0].contains(Q(13,12)))
        self.assertTrue(BA[0][0].contains(1))
        self.assertFalse(AB[0][0].contains(1))

    def test_K4_keeps_degree5_not_degree4(self):
        x=Q(1,3)
        got=c.finite_polynomial([[x]],K=4)[0][0]
        expected=sum((x**k/math.factorial(k) for k in range(6)),Q(0))
        self.assertEqual(got,expected)
        self.assertNotEqual(got,sum((x**k/math.factorial(k) for k in range(5)),Q(0)))

    def test_error_transport_keeps_nonunitary_amplification(self):
        self.assertEqual(c.product_error_bound([2,3],[Q(1,10),Q(1,5)]),Q(18,25))

    def test_projection_without_invariance_changes_operator(self):
        X=[[0,1],[1,0]];N=[[0,0],[0,1]]
        self.assertNotEqual(c.multiply(X,N),c.multiply(N,X))

    def test_state_norm_defect_cannot_be_silently_ignored(self):
        psi=[Q(3,5),Q(4,5)+Q(1,1000)]
        n=sum(x*x for x in psi)
        self.assertNotEqual(n,1)
        ray_signal=(psi[0]**2-psi[1]**2)/n
        encoded_signal=psi[0]**2-psi[1]**2
        self.assertNotEqual(ray_signal,encoded_signal)


if __name__=='__main__':unittest.main(verbosity=2)
