"""Artificial numerical equivalence only; no Qiskit/compiler or real NPZ."""
import math,unittest
import numpy as np
from scipy.linalg import expm,logm
from trottertracks.resource_applicability.h4_geometry import gaussian_structure_proposal as proposal
from trottertracks.resource_applicability.h4_geometry.identity import Stop


def assert_allclose(actual,expected,*,atol,rtol):
    """Avoid numpy.testing's subprocess-based CPU detection; no children."""
    a,b=np.asarray(actual),np.asarray(expected)
    assert a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    error=float(np.max(np.abs(a-b)))
    assert np.all(np.abs(a-b)<=atol+rtol*np.abs(b)), 'maximum absolute error: '+str(error)


def assert_array_equal(actual,expected):
    assert np.array_equal(actual,expected)


def lift_by_minors(U):
    n=len(U);dim=1<<n;matrix=np.zeros((dim,dim),dtype=complex)
    occupancies=[[i for i in range(n) if bits>>i&1] for bits in range(dim)]
    for row,bra in enumerate(occupancies):
        for column,ket in enumerate(occupancies):
            if len(bra)==len(ket):matrix[row,column]=1. if not bra else np.linalg.det(U[np.ix_(bra,ket)])
    return matrix


def plan_operator(plan):
    n=plan['modes'];dim=1<<n
    diagonal=np.array([np.exp(1j*sum(plan['phases'][i] for i in range(n) if bits>>i&1)) for bits in range(dim)])
    result=np.diag(diagonal)
    for upper,lower,G in plan['rotations']:
        pair=proposal.fermionic_pair_matrix(G);next_result=np.zeros_like(result);mask=(1<<upper)|(1<<lower)
        for old in range(dim):
            local=(old>>upper&1)|((old>>lower&1)<<1)
            for new in range(4):
                target=(old&~mask)|((new&1)<<upper)|((new>>1)<<lower)
                next_result[target,:]+=pair[new,local]*result[old,:]
        result=next_result
    return result


def old_dense_operator(U):
    """Independent little-endian JW dGamma(log U), same dense fallback math."""
    n=len(U);dim=1<<n;anti=logm(U);anti=(anti-anti.conj().T)*.5;generator=np.zeros((dim,dim),dtype=complex)
    for column in range(dim):
        for q in range(n):
            if not (column>>q&1):continue
            removed=column^(1<<q);annihilation_sign=(-1)**((column&((1<<q)-1)).bit_count())
            for p in range(n):
                if removed>>p&1:continue
                creation_sign=(-1)**((removed&((1<<p)-1)).bit_count())
                generator[removed|(1<<p),column]+=anti[p,q]*annihilation_sign*creation_sign
    return expm(generator)


def artificial_unitary(n,seed=17):
    rng=np.random.default_rng(seed);matrix=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));Q,R=np.linalg.qr(matrix)
    return Q@np.diag(np.diag(R)/np.abs(np.diag(R)))


class GaussianStructureTests(unittest.TestCase):
    def compare(self,U,dense=True):
        plan=proposal.givens_plan(U);new=plan_operator(plan)
        assert_allclose(new,lift_by_minors(U),atol=1e-12,rtol=0)
        if dense:assert_allclose(new,old_dense_operator(U),atol=1e-12,rtol=0)
        self.assertEqual(new[0,0],1.)
        assert_allclose(new.conj().T@new,np.eye(len(new)),atol=1e-12,rtol=0)
        self.assertLessEqual(len(plan['rotations']),len(U)*(len(U)-1)//2)
        return plan,new
    def test_single_mode_phase_vacuum_and_branch_phase(self):
        for phase in (.31,-.17,math.pi):
            plan,new=self.compare(np.array([[np.exp(1j*phase)]]))
            assert_allclose(new,np.diag([1,np.exp(1j*phase)]),atol=1e-12,rtol=0)
    def test_identity_signed_zero_no_spurious_rotations(self):
        U=np.eye(8,dtype=complex);U[0,1]=complex(-0.,0.)
        plan,new=self.compare(U)
        self.assertEqual(len(plan['rotations']),0);self.assertEqual(plan['phases'],(0.,)*8)
    def test_complex_random2_and3_modes_all_fock_sectors(self):
        for n in (2,3):self.compare(artificial_unitary(n))
    def test_complex_random4_and5_modes_dense_legacy(self):
        for n in (4,5):self.compare(artificial_unitary(n,29))
    def test8_modes_256_by256_dense_old_new_equality(self):
        plan,new=self.compare(artificial_unitary(8,41))
        self.assertEqual(new.shape,(256,256));self.assertEqual(len(plan['rotations']),28)
    def test_real_orthogonal_negative_determinant(self):
        rng=np.random.default_rng(43);U=np.linalg.qr(rng.normal(size=(4,4)))[0]
        U[:,0]*=np.sign(np.linalg.det(U))*-1;self.compare(U)
    def test_permutation_with_fermionic_double_occupancy_sign(self):
        U=np.array([[0,1],[1,0]],complex);plan,new=self.compare(U)
        self.assertAlmostEqual(new[3,3],-1.)
    def test_phase_branch_minus_identity(self):
        self.compare(-np.eye(4,dtype=complex))
    def test_adjacent_pairs_reconstruction_and_input_unchanged(self):
        U=artificial_unitary(8);before=U.copy();plan=proposal.givens_plan(U)
        self.assertTrue(all(j==i+1 and not G.flags.writeable for i,j,G in plan['rotations']))
        assert_array_equal(U,before);assert_allclose(proposal.orbital_matrix(plan),U,atol=1e-12,rtol=0)
    def test_controlled_cosine_sine_unitary_preserves_relative_phase(self):
        U=artificial_unitary(3,47);plan,new=self.compare(U);old=old_dense_operator(U);dim=len(new)
        h=np.array([[1,1],[1,-1]],complex)/np.sqrt(2);H=np.kron(h,np.eye(dim));S=np.kron(np.diag([1,-1j]),np.eye(dim))
        def controlled(G):
            C=np.eye(2*dim,dtype=complex);C[dim:,dim:]=G;return C
        for axis in ('cosine','sine'):
            middle=np.eye(2*dim) if axis=='cosine' else S
            assert_allclose(H@middle@controlled(new)@H,H@middle@controlled(old)@H,atol=1e-12,rtol=0)
    def test_invalid_nonfinite_nonunitary_and_size_rejected(self):
        for U in (np.eye(9),np.zeros((2,2)),np.ones((2,3)),np.array([[float('nan')]]),np.array([[float('inf')]]),np.eye(2)*1.0001):
            with self.assertRaises(Stop):proposal.givens_plan(U)
    def test_inactive_policy_and_no_unchanged_cost_claim(self):
        for name in ('approved','runtime_authorization','production_wiring_present'):self.assertIs(proposal.POLICY[name],False)
        self.assertIs(proposal.POLICY['cost_semantics_change'],True)
        self.assertIs(proposal.POLICY['compiled_gate_metrics_equivalent_to_dense'],False)
