"""Artificial laws/Paulis only; no snapshot, preparation, RNG or quantum builder."""
from fractions import Fraction as F
from itertools import product
import math
from types import SimpleNamespace as NS
from unittest import TestCase, main

import numpy as np

from trottertracks.algorithm_codesign.a1_connections import SamplerConnection
from trottertracks.algorithm_codesign.a1_control_variates import CategoricalLaw, OccurrenceLaw
from trottertracks.algorithm_codesign.a1_law_validation import state_free_bias, fraction_record


def connection(qn=(F(4,5), F(1,5)), ql=(F(1,4), F(3,4))):
    law = OccurrenceLaw(CategoricalLaw((0,2), qn, "invented-order"),
                        CategoricalLaw(("a","b"), ql, "invented-label"))
    return SamplerConnection(law, (), (), (), (), 1.)


class A1LawBounds(TestCase):
    def test_bound_contains_invented_noncommuting_operator_difference(self):
        # This is a tiny artificial algebra check, never a molecular state.
        c = connection(); tau = .2
        b = math.hypot(1.,tau) + tau*tau/2*math.hypot(1.,tau/3)
        bound = state_free_bias(c, (.375,.625), lambda_r=1., tau=tau, normalization=b)
        I = np.eye(2,dtype=complex)
        P = (np.array([[0,1],[1,0]],dtype=complex), np.diag([1.,-1.]).astype(complex))
        observed = np.zeros((2,2),dtype=complex)
        for n, qn in zip((0,2), c.law.orders.masses):
            angle = math.atan(tau/(n+1))
            for indices in product(range(2),repeat=n+1):
                rotation = math.cos(angle)*I - 1j*math.sin(angle)*P[indices[0]]
                products = I.copy()
                for j in indices[1:]: products = P[j] @ products
                mass = qn
                for j in indices: mass *= c.law.labels.masses[j]
                observed += float(mass)*(-1)**(n//2)*(rotation @ products)
        A = -1j*tau*(.375*P[0]+.625*P[1])
        target = (I+A+(A@A)/2+(A@A@A)/6)/b
        actual = np.linalg.norm(observed-target,2)
        self.assertLessEqual(actual,float(F(bound["conditional_coordinate_bias_upper"]["exact"])))
        self.assertFalse(bound["full_numerical_bias_closed"])

    def test_source_scalar_rounding_and_normalizer_are_not_silently_zero(self):
        b = state_free_bias(connection(), (.3,.7), lambda_r=1.0000000000000002,
                            tau=.125, normalization=1.)
        self.assertGreater(F(b["normalization_bias_upper"]["exact"]),0)
        self.assertNotEqual(F(b["reference_s"]),F.from_float(.125))
        self.assertGreater(F(b["conditional_coordinate_bias_upper"]["exact"]),0)

    def test_invalid_coefficient_or_scalar_scope_rejected(self):
        for coefficients,lam in (((0.,1.),1.), ((.2,),1.), ((.3,.7),0.)):
            with self.subTest(coefficients=coefficients,lam=lam),self.assertRaises(ValueError):
                state_free_bias(connection(),coefficients,lambda_r=lam,tau=.2,normalization=1.)

    def test_float_preview_is_outward_and_exact_value_is_preserved(self):
        q = F(1,10); r = fraction_record(q)
        self.assertEqual(F(r["exact"]),q)
        self.assertGreaterEqual(F.from_float(r["float_upper_preview"]),q)


if __name__ == "__main__":
    main()
