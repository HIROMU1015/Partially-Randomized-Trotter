"""Registered-outside synthetic semantics only; no molecular/compile/eigen calls."""
from fractions import Fraction as F
import math
import unittest
from unittest.mock import patch
import numpy as np

from trottertracks.representation_exploration.qc_relative_block_v1 import (
    detuning_plan, affine_value, parity_plan, parity_sign, encode_pair,
    valid_encoded, relative_matrix, angle_recipe, precision_bound, quantize, structural_ir)


def ladder(x, create, annihilate):
    # Independent occupation ladder reference: operator applied right to left.
    phase = 1
    for i, creation in [(i, False) for i in reversed(annihilate)] + [(i, True) for i in reversed(create)]:
        bit = (x >> i) & 1
        if bit == int(creation):
            return None, 0
        phase *= (-1) ** sum((x >> k) & 1 for k in range(i))
        x ^= 1 << i
    return x, phase


def exponential(matrix):
    # Scaling + Taylor powers, intentionally no eig/SVD/SciPy expm.
    scale = max(0, math.ceil(math.log2(max(1, np.max(np.sum(np.abs(matrix), axis=1))))))
    z = matrix / 2**scale; result = np.eye(len(z), dtype=complex); term = result.copy()
    for k in range(1, 81):
        term = term @ z / k; result += term
    for _ in range(scale):
        result = result @ result
    return result


def energy(x, eps, J):
    return sum(eps[i] * ((x >> i) & 1) for i in range(len(eps))) + sum(J[i][j] * ((x >> i) & 1) * ((x >> j) & 1) for i in range(len(eps)) for j in range(i+1, len(eps)))


def fixture(n):
    eps = [F((-1)**i * (i+1), 13) for i in range(n)]
    J = [[F(0) if i == j else F((-1)**(i+j) * (i+j+1), 29) for j in range(n)] for i in range(n)]
    return eps, J


class Synthetic(unittest.TestCase):
    def setUp(self):
        self.guards = [patch.object(np.linalg, name, side_effect=AssertionError('EIG_OR_SVD_FORBIDDEN')) for name in ('eig', 'eigh', 'eigvals', 'eigvalsh', 'svd')]
        for g in self.guards:
            g.start()
    def tearDown(self):
        for g in self.guards:
            g.stop()

    def test_affine_signed_single_double(self):
        for create, annihilate in (((0,), (3,)), ((1, 4), (3, 0)), ((4, 1), (0, 3))):
            n = 5; eps, J = fixture(n); plan = detuning_plan(eps, J, create, annihilate)
            for x in range(1 << n):
                y, sign = ladder(x, create, annihilate)
                if sign:
                    self.assertEqual(affine_value(plan, x), energy(y, eps, J) - energy(x, eps, J))

    def test_parity_against_ladder(self):
        for create, annihilate in (((0,), (4,)), ((0, 3), (4, 1)), ((3, 0), (1, 4))):
            plan = parity_plan(5, create, annihilate)
            for x in range(32):
                y, sign = ladder(x, create, annihilate)
                if sign:
                    self.assertEqual(parity_sign(plan, x), sign)
                    self.assertEqual(parity_sign(plan, y), sign)

    def test_pair_encode_valid_decode(self):
        for create, annihilate in (((0,), (4,)), ((0, 3), (4, 1)), ((4, 1), (3, 0))):
            support = create + annihilate
            for x in range(32):
                enc = encode_pair(5, create, annihilate, x)
                self.assertEqual(encode_pair(5, create, annihilate, enc, inverse=True), x)
                source = all(((x >> i) & 1) == int(i in annihilate) for i in support)
                target = all(((x >> i) & 1) == int(i in create) for i in support)
                self.assertEqual(valid_encoded(5, create, annihilate, enc), source or target)

    def test_full_fock_independent_reference(self):
        n = 5; eps, J = fixture(n)
        for create, annihilate, assist in (((0,), (4,), None), ((1, 4), (3, 0), None), ((0,), (4,), 2)):
            plan = detuning_plan(eps, J, create, annihilate); pp = parity_plan(n, create, annihilate)
            for t in (.17, -.23):
                A = np.diag([float(energy(x, eps, J)) for x in range(32)])
                B = np.zeros((32, 32))
                U = np.eye(32, dtype=complex)
                for x in range(32):
                    y, sign = ladder(x, create, annihilate)
                    if not sign:
                        continue
                    g = .19 * (1 if assist is None else (x >> assist) & 1)
                    B[x, y] = B[y, x] = g * sign
                    block = np.asarray(relative_matrix(float(affine_value(plan, x)), g * parity_sign(pp, x), t))
                    U[np.ix_([y, x], [y, x])] = block
                reference = np.diag(np.exp(1j * t * np.diag(A))) @ exponential(-1j * t * (A+B))
                self.assertLess(np.max(np.sum(np.abs(U-reference), axis=1)), 3e-13)
                for x in range(32):
                    for y in range(32):
                        if x.bit_count() != y.bit_count():
                            self.assertEqual(U[x,y], 0)

    def test_angles_zero_detuning_signed_negative_time(self):
        def rotation(axis, x):
            if axis == 'RZ':
                return np.diag([np.exp(-.5j*x),np.exp(.5j*x)])
            return np.array([[math.cos(x/2),-math.sin(x/2)],[math.sin(x/2),math.cos(x/2)]])
        for d, b, t in ((0,.3,.2),(-.4,.3,-.2),(.01,-.3,.2),(1e-12,.3,.2),(-1,.1,1)):
            out = np.eye(2, dtype=complex)
            for axis, angle in angle_recipe(d,b,t)['chronological_rotations']:
                out = rotation(axis,angle) @ out
            np.testing.assert_allclose(out, relative_matrix(d,b,t),atol=3e-15,rtol=0)

    def test_zero_coupling_skip(self):
        for d in (-100., 0., 100.):
            self.assertEqual(relative_matrix(d,0,10),((1+0j,0j),(0j,1+0j)))
            self.assertEqual(angle_recipe(d,0,10)['chronological_rotations'],[])

    def test_coeff_rounding_bound(self):
        for d,b,t in ((-.3,.2,-.7),(0,.2,.7),(.3,-.2,.7)):
            dt = float(quantize(F(str(d)),6)); bt = float(quantize(F(str(b)),6))
            diff=np.asarray(relative_matrix(d,b,t))-np.asarray(relative_matrix(dt,bt,t))
            # Difference of SU(2) matrices has equal singular values; closed norm.
            norm=math.sqrt(float(np.sum(np.abs(diff)**2))/2)
            self.assertLessEqual(norm,precision_bound(t,abs(b),abs(d-dt),abs(b-bt))+3e-15)

    def test_fixed_point_coefficient_bound(self):
        eps,J=fixture(5);plan=detuning_plan(eps,J,(0,),(4,));f=8
        rounded={'constant':quantize(plan['constant'],f),'spectator_weights':{i:quantize(v,f) for i,v in plan['spectator_weights'].items()}}
        budget=F(1+len(plan['spectator_weights']),2**(f+1))
        for x in range(32):
            self.assertLessEqual(abs(affine_value(plan,x)-affine_value(rounded,x)),budget)
        self.assertEqual(quantize(F(-1,2),0),0)

    def test_control_and_average_phase(self):
        d,b,t=.3,.2,.7;H=np.array([[.61+d/2,b],[b,.61-d/2]])
        U=np.asarray(relative_matrix(d,b,t));ref=np.diag(np.exp(1j*t*np.diag(H)))@exponential(-1j*t*H)
        np.testing.assert_allclose(U,ref,atol=3e-15,rtol=0)
        ctrl=np.zeros((4,4),complex);ctrl[:2,:2]=np.eye(2);ctrl[2:,2:]=U
        expected=np.zeros((4,4),complex);expected[:2,:2]=np.eye(2);expected[2:,2:]=ref
        np.testing.assert_allclose(ctrl,expected,atol=3e-15,rtol=0)
        self.assertGreater(np.max(np.abs(ctrl-np.exp(.1j)*ctrl)),.05)

    def test_adjoint_order(self):
        C=np.asarray(relative_matrix(.7,.3,.4));minus=np.asarray(relative_matrix(.7,.3,-.4))
        self.assertGreater(np.max(np.abs(minus-C.conj().T)),.01)
        np.testing.assert_allclose(C.conj().T@C,np.eye(2),atol=3e-15,rtol=0)

    def test_uncompute_invariants(self):
        pp=parity_plan(5,(0,3),(4,1));plan=detuning_plan(*fixture(5),(0,3),(4,1))
        for x in range(32):
            y,sign=ladder(x,(0,3),(4,1))
            if sign:
                self.assertEqual(affine_value(plan,x),affine_value(plan,y))
                self.assertEqual(parity_sign(pp,x),parity_sign(pp,y))
                ex,ey=(encode_pair(5,(0,3),(4,1),z) for z in (x,y))
                self.assertEqual(ex^ey,1)

    def test_invalid_domain(self):
        eps,J=fixture(5)
        for create,ann in (((0,),(0,)),((True,),(4,)),((0,),(5,)),((0,1),(4,))):
            with self.assertRaises(ValueError):
                detuning_plan(eps,J,create,ann)
        with self.assertRaises(ValueError):
            precision_bound(.2,.1,-.1,0)
        with self.assertRaises(ValueError):
            relative_matrix(float('nan'),.1,.2)

    def test_structural_cost_not_measured(self):
        ir=structural_ir(8,(0,3),(7,4))
        self.assertEqual(ir['spectator_coefficient_count'],4)
        self.assertIsNone(ir['compiled_T_count'])
        self.assertFalse(ir['reversible_function_evaluator_implemented'])


if __name__ == '__main__':
    unittest.main()
