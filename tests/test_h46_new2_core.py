"""Artificial correctness tests only; no molecular inputs or compile."""
import math
import unittest
import numpy as np
from scipy.linalg import expm
from h46_new2_core import weights, allocation, spectrum, polynomial, compare_prefix, matrix_only_tail_coefficients


class New2Tests(unittest.TestCase):
    def setUp(self):
        self.a=np.array([[.7,.2+.1j],[.2-.1j,-.1]])
        self.b=np.array([[.15,-.3j],[.3j,-.2]])
        self.state=np.array([1.,1.j])/math.sqrt(2)

    def test_registered_coefficients_and_tail_step_accounting(self):
        self.assertEqual(len(weights('4th(new_2)')),5)
        for label in ('2nd','4th','4th(new_2)'):
            w=weights(label);self.assertAlmostEqual(sum(w),1.,places=14)
            self.assertEqual(w,tuple(reversed(w)))
            r=allocation(w,32);self.assertEqual(sum(r),32);self.assertGreaterEqual(min(r),1)
            self.assertEqual(allocation(w,32),r)

    def test_invalid_budget_and_nonhermitian_rejected(self):
        with self.assertRaises(ValueError):allocation(weights('4th(new_2)'),4)
        with self.assertRaises(ValueError):spectrum(np.array([[0,1],[0,0]]))
        with self.assertRaises(ValueError):polynomial([1],.2,3)

    def test_negative_time_matches_dense_Taylor_and_adjoint(self):
        ev,v,vh=spectrum(self.b)
        from trotterlib.rte import finite_taylor_operator
        for k in (2,4):
            p=v@(polynomial(ev,-.2,k)[:,None]*vh)
            np.testing.assert_allclose(p,finite_taylor_operator(self.b,-.2,k),atol=1e-13)
            pos=v@(polynomial(ev,.2,k)[:,None]*vh)
            np.testing.assert_allclose(p,pos.conj().T,atol=1e-13)

    def test_new2_practical_fourth_order_and_s2_second_order(self):
        errors={}
        for label in ('2nd','4th','4th(new_2)'):
            e=[]
            for q in (2,4,8):
                out=compare_prefix(self.state,[spectrum(self.a)],spectrum(self.b),0.,1.,label,q,T=.5)
                e.append(np.linalg.norm(out['exact_tail_vector']-expm(-.5j*(self.a+self.b))@self.state))
            orders=[math.log(e[i]/e[i+1],2) for i in (0,1)]
            for order in orders:self.assertLess(abs(order-(2 if label=='2nd' else 4)),.4)
            errors[label]=e
        self.assertLess(errors['4th(new_2)'][-1],errors['2nd'][-1])

    def test_scalar_phase_is_global_and_reference_not_rotated(self):
        args=(self.state,[spectrum(self.a)],spectrum(self.b))
        a=compare_prefix(*args,0.,1.,'4th(new_2)',2,T=.5)
        b=compare_prefix(*args,.73,1.,'4th(new_2)',2,T=.5)
        np.testing.assert_allclose(b['exact_tail_vector'],np.exp(-.5j*.73)*a['exact_tail_vector'],atol=1e-13)
        for x,y in zip(a['finite_rows'],b['finite_rows']):
            self.assertLess(y['raw_corrected_vector_discrepancy'],1e-13)
            self.assertAlmostEqual(x['log_B'],y['log_B'])

    def test_full_fock_scalar_extraction_without_circuit_basis(self):
        from types import SimpleNamespace
        ham=SimpleNamespace(n_blocks=1,g_matrices=[np.diag([.3,-.7])],lambdas=[.8])
        scalar,lam=matrix_only_tail_coefficients(ham,0)
        energies=np.array([0.,.8*.3**2,.8*(-.7)**2,.8*(.3-.7)**2])
        self.assertAlmostEqual(scalar,float(energies.mean()),places=14)
        self.assertGreater(lam,0)
        with self.assertRaises(ValueError):matrix_only_tail_coefficients(ham,1)


if __name__=='__main__':unittest.main()
