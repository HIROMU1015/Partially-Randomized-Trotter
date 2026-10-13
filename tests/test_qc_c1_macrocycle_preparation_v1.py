"""Off-domain exact algebra/schema/gate tests; no registered coefficient substitution."""
import ast
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import unittest
from trottertracks.representation_exploration import c1_macrocycle_algebra_v1 as alg
from trottertracks.representation_exploration import c1_macrocycle_contract_v1 as rules
from trottertracks.representation_exploration.c1_macrocycle_execution_v1 import (
    LaunchError,check_approval,SCOPE,OUTPUT)

ROOT=Path(__file__).resolve().parents[1]


class Preparation(unittest.TestCase):
    def test_recursive_BCH_and_fixed_macros(self):
        self.assertEqual(alg.preparation_proofs()['macro_catalog_count'],24)

    def test_reverse_pair_cyclic_conjugation(self):
        for order in alg.REPRESENTATIVES:
            x,y,z=order
            halves=[(x,F(1,2)),(y,F(1,2)),(z,F(1,2))]
            forward=halves+list(reversed(halves))
            reverse=list(reversed(halves))+halves
            self.assertEqual(alg.canonical_stages(forward),alg.kernel(order))
            self.assertEqual(alg.canonical_stages(reverse),alg.kernel(tuple(reversed(order))))

    def test_subdivision_cubic_scaling(self):
        for m in (2,3,4):
            stages=[(n,t/F(m)) for _ in range(m) for n,t in alg.kernel(('A','B','C'))]
            cubic={w:-v for w,v in alg.log_series(stages,3).items() if len(w)==3}
            self.assertEqual(cubic,alg.scale(alg.kernel_error(('A','B','C')),F(1,m*m)))

    def test_pauli_phase_exact(self):
        self.assertEqual(rules.pauli_product('XI','YI'),('ZI',1))
        self.assertEqual(rules.pauli_product('YI','XI'),('ZI',3))
        self.assertEqual(rules.pauli_product('XX','YY'),('ZZ',2))

    def test_exact_hermitian_substitution(self):
        # Different off-domain symbolic fixture, never a pilot input.
        poly=alg.commutator(alg.symbol('A'),alg.commutator(alg.symbol('A'),alg.symbol('B')))
        self.assertEqual(rules.substitute(poly,{'A':{'Z':F(2)},'B':{'X':F(3)},'C':{'Y':F(5)}}),{'X':F(48)})

    def test_nonhermitian_rejected(self):
        with self.assertRaises(ValueError):
            rules.substitute({('A','B'):F(1)},{'A':{'X':F(1)},'B':{'Y':F(1)},'C':{'Z':F(1)}})

    def test_joint_norm_cancellation(self):
        # Shared perturbation can cancel exactly; independent boxes cannot.
        d1={'Z':F(1)}; d2={'Z':F(-1)}
        merged={'Z':F(0)}
        self.assertLess(rules.l1(merged),rules.l1(d1)+rules.l1(d2))

    def test_positive_time_and_invalid_catalog(self):
        for row in alg.catalog():
            self.assertTrue(all(t>0 for n,t in alg.macrocycle(row['j'],row['k'],row['a'])))
        for args in ((0,0,F(1,4)),(0,1,F(1,5)),(3,0,F(1,4))):
            with self.assertRaises(ValueError): alg.macrocycle(*args)

    def test_controlled_identity_phase(self):
        counts=rules.native_counts([('A',F(1))],{'A':{'I':F(7)}})
        self.assertEqual((counts['RZ'],counts['CX']),(1,0))

    def test_single_gets_same_sharing(self):
        frags={'A':{'Z':F(1)},'B':{'X':F(1)},'C':{'Y':F(1)}}
        count=rules.native_counts(alg.kernel(('A','B','C')),frags)
        self.assertEqual(count['RZ'],8) # four steady stages, not five.
        for row in alg.catalog():
            raw=alg.macrocycle(row['j'],row['k'],row['a'])
            self.assertGreaterEqual(rules.native_counts(raw,frags)['RZ'],3*count['RZ'])

    def test_no_noncommuting_reorder(self):
        stages=[('A',F(1)),('B',F(1)),('A',F(1))]
        self.assertEqual(alg.canonical_stages(stages),tuple(stages))

    def test_oracle_input_rejected_first(self):
        for field in ('ground_energy','true_gap','ground_vector','selected_by_reference'):
            with self.assertRaisesRegex(ValueError,'INPUT_FIELDS_OR_ORACLE'):
                rules.validate_input({field:0})

    def test_information_state_overlap_formula(self):
        self.assertEqual(rules.expectation({'ZZ':F(2),'XI':F(99)},'00'),2)
        self.assertEqual(rules.expectation({'ZI':F(2)},'11'),-2)

    def test_rational_remainder_no_gap_rejection(self):
        self.assertIsNone(rules.analytic_bound(F(1),None,F(1),F(1),0))
        bound=rules.analytic_bound(F(1),F(2),F(1),F(1,2),0)
        self.assertIsInstance(bound,F)
        self.assertEqual(bound,F(1,128)+F(1,4096)*(F(2,3)*4**5+1))

    def test_missing_draft_and_wrong_scope_approval(self):
        good={'type':'USER_EXPLICIT_SCIENCE_APPROVAL','user_approval_reference':'test-fixture-only',
              'scope_id':SCOPE,'attempts':1,'retries':0,'output':OUTPUT,'x':'locked'}
        for bad in (None,{},dict(good,type='DRAFT'),dict(good,retries=1),dict(good,scope_id='old_scope')):
            with self.assertRaises(LaunchError): check_approval(bad,{'x':'locked'})
        with self.assertRaises(LaunchError): check_approval(good,{'x':'different'})

    def test_no_approval_entry_rejects_without_output_or_numpy(self):
        code=('import runpy,sys;sys.argv=["runner"];'
              'try:\n runpy.run_path("scripts/run_qc_c1_macrocycle_pilot_v1.py",run_name="__main__")\n'
              'except ValueError as e:\n assert "SEPARATE_EXPLICIT" in str(e); assert "numpy" not in sys.modules\n'
              'else:\n raise AssertionError("gate failed")')
        # Use a real multiline script; no quantum/scientific call can pass the gate.
        code=code.replace(';try:\n','\ntry:\n')
        run=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertFalse((ROOT/OUTPUT).exists())

    def test_all_new_sources_syntax_only(self):
        names=list((ROOT/'src/trottertracks/representation_exploration').glob('c1_macrocycle_*_v1.py'))
        names+=list((ROOT/'scripts').glob('*qc_c1_macrocycle*_v1.py'))
        for name in names: ast.parse(name.read_text(),filename=str(name))


if __name__=='__main__': unittest.main()
