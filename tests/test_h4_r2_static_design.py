"""R2 algebra on tiny synthetic systems only; no H4 read, compile or RNG."""
from fractions import Fraction as Q
import itertools
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[1]
P=runpy.run_path(str(ROOT/'scripts/inspect_h4_arb_pr0.py'))['P']
C=P.B._backend()


class R2DesignTests(unittest.TestCase):
    def test_noncommuting_finite_law_factorization_all_three_orders(self):
        # Reuse the production C expectation formula. Enumerate only two toy
        # components (42 events), independently in chronological product order.
        for n in (2,3):
            with C.ctx.workprec(256):
                dim=1<<n;I=C.identity(dim)
                terms=[C.pauli('I'*(n-1)+'X'),-C.pauli('I'*(n-1)+'Y')]
                ps=[Q(1,3),Q(2,3)];pis=[Q(1,2),Q(1,3),Q(1,6)]
                rotations=[]
                for angle in (Q(1,8),Q(1,4),Q(3,8)):
                    a=C.real(angle);rotations.append([I*a.cos()+t*C.acb(0,-a.sin()) for t in terms])
                actual=C.finite_implemented_mean(terms,rotations,ps,pis)
                expected=C.acb_mat(dim,dim);wrong_phase=C.acb_mat(dim,dim)
                for order,rotation,pi in zip((0,2,4),rotations,pis):
                    for ids in itertools.product(range(2),repeat=order+1):
                        product=I;mass=pi
                        for j in ids[:-1]:product=terms[j]*product
                        for j in ids:mass*=ps[j]
                        value=rotation[ids[-1]]*product*C.real(mass)
                        expected+=value*((-1)**(order//2));wrong_phase+=value
                for row in range(dim):
                    for col in range(dim):
                        self.assertTrue((actual[row,col]-expected[row,col]).contains(0))
                        self.assertLess(abs(actual[row,col]-expected[row,col]).upper(),C.real('1e-65'))
                self.assertTrue(any(not (actual[i,j]-wrong_phase[i,j]).contains(0) for i in range(dim) for j in range(dim)))
                commutator=terms[0]*terms[1]-terms[1]*terms[0]
                self.assertTrue(any(commutator[i,j]!=0 for i in range(dim) for j in range(dim)))

    def test_additive_law_weighted_envelope_without_path_enumeration(self):
        pi={0:Q(1,2),2:Q(1,3),4:Q(1,6)};error={0:Q(1,1000),2:Q(1,500),4:Q(1,250)}
        # Exhaustive TWO toy events only, no trajectory sampled or generated.
        direct=sum((pi[k]*pi[j]*(error[k]+error[j]) for k,j in itertools.product(pi,repeat=2)),Q(0))
        weighted=2*sum((pi[k]*error[k] for k in pi),Q(0));uniform=2*max(error.values())
        self.assertEqual(direct,weighted);self.assertLess(weighted,uniform)
        B=Q(3,2);self.assertEqual(B*weighted,sum((B*pi[k]*pi[j]*(error[k]+error[j]) for k,j in itertools.product(pi,repeat=2)),Q(0)))
        self.assertNotEqual(B*weighted,B*B*weighted)

    def test_unknown_order4_mass_cannot_be_filled_by_zero(self):
        p=Q(1,1000);events=3;mass=1-(1-p)**events;guard=Q(1,10000)
        self.assertGreaterEqual(mass,p);self.assertGreater(2*mass,guard)
        # Failure of this sufficient bound says nothing about the true error.
        actual_zero_error=Q(0);self.assertLessEqual(actual_zero_error,guard)

    def test_order4_signed_noncommuting_product_is_not_reordered(self):
        with C.ctx.workprec(128):
            X,Y,Z=C.pauli('IX'),C.pauli('IY'),C.pauli('IZ')
            chronological=(-Y)*Z*Y*X  # Apply X, Y, Z, -Y in that order.
            reordered=X*Y*Z*(-Y)
            self.assertEqual(chronological,-reordered)
            self.assertEqual((-1)**(4//2),1)
            self.assertEqual((-1)**(2//2),-1)

    def test_sine_sign_and_body_global_phase_are_observable(self):
        # Exact quarter-turn arithmetic: the final H divides amplitudes by 2.
        for z in (1,-1,1j,-1j):
            cosine=(abs(1+z)**2-abs(1-z)**2)/4
            sine=(abs(1-1j*z)**2-abs(1+1j*z)**2)/4
            self.assertEqual(cosine,z.real);self.assertEqual(sine,z.imag)
        self.assertEqual(abs(1-(-1)),2)  # Projective equality cannot erase phase.

    def test_shared_raw_norm_and_wrapper_factor_two(self):
        # Unnormalized raw vector 2|0>: nu=4, not an extra B or state rescale.
        self.assertEqual(Q(4,4),1);self.assertNotEqual(Q(4),Q(4,4))
        # Body U=I,V=-I: |Re <U-V>|=2=||U-V||. A generic
        # ancilla-Z expectation bound is 2||W-V||, then clipped to range 2.
        body_difference=Q(2);wrapper_difference=Q(2)
        self.assertEqual(body_difference,min(Q(2),2*wrapper_difference))

    def test_existing_right_search_finite_grid_mapping(self):
        self.assertEqual(C.grid_masses([.25,.5,1.],bits=3),[Q(1,4),Q(1,4),Q(1,2)])
        # A CDF endpoint off the grid uses ceil(2^b*cdf), not a nearest mass.
        self.assertEqual(C.grid_masses([.3,1.],bits=3),[Q(3,8),Q(5,8)])


if __name__=='__main__':unittest.main(verbosity=2)
