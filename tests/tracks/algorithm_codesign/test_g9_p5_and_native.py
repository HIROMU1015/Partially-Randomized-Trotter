import unittest
from fractions import Fraction as F
import numpy as np
from trottertracks.algorithm_codesign.g9_p5 import P5Closed,audit_p5
from trottertracks.algorithm_codesign.g9_native import (cts_events,exact_target,native_ir,CQ,V,cpauli,simplify,adjoint,cost,provider_polynomials,poly_product,A)
from trottertracks.algorithm_codesign.g9_matrix import circuit,Q_matrices,circuit_error,controlled,event_operator
from trottertracks.algorithm_codesign.g7_generator import DeterministicBits,FullReturnGenerator


class G9(unittest.TestCase):
    def test_formal_off_domain_two(self):
        self.assertTrue(audit_p5((F(2,9),F(7,9)),F(3,5))['formal_coefficients_exact'])
    def test_formal_off_domain_three(self):
        self.assertTrue(audit_p5((F(1,7),F(2,7),F(4,7)),F(1,3))['formal_coefficients_exact'])
    def test_single_label(self):
        self.assertEqual(audit_p5((F(1),),F(1,3))['groups'],1)
    def test_bad_domain(self):
        with self.assertRaises(ValueError):P5Closed((F(1,2),F(1,2)),F(2))
    def test_same_ideal_Green_coefficients(self):
        p=(F(1,7),F(2,7),F(4,7));g=P5Closed(p,F(1,3));old=FullReturnGenerator(p,F(1,3),5)
        for e in g.reference_events():
            par=old.kernel.parent(e['word']);a,s,b=g.parent_coefficients(e['word'])
            self.assertEqual((par.a,par.s),(a,s))
            self.assertEqual(par.masses[par.children.index(e['child'])],b[e['child']])
    def test_stream_no_reference_call(self):
        g=P5Closed((F(2,9),F(7,9)),F(1,3));g.reference_events=lambda:(_ for _ in ()).throw(AssertionError())
        self.assertGreater(g.sample(DeterministicBits('off-domain-G9'))['proposal'],0)
    def test_each_weight_exact(self):
        g=P5Closed((F(2,9),F(7,9)),F(1,3))
        self.assertTrue(all(e['proposal']*e['weight']==e['coefficient'] for e in g.reference_events()))
    def test_completion_not_stack_surrogate(self):
        g=P5Closed((F(2,9),F(7,9)),F(1,3));self.assertEqual(sum(g.r4),2*F(2,9)**2*F(7,9)**2)
    def test_group_build_no_triple_label_table(self):
        from unittest.mock import patch
        with patch.object(P5Closed,'level2',side_effect=AssertionError('child table during build')):
            g=P5Closed((F(1,10),F(2,10),F(3,10),F(4,10)),F(1,3))
            self.assertEqual(len(g.groups),17)
    def test_provider_exact_symbolic_involutions(self):
        for q in provider_polynomials():self.assertEqual(poly_product(q,q),{('III',0):A(1)})
    def test_physical_CQ_direct_phase(self):
        for i,q in enumerate(Q_matrices()):self.assertLess(np.linalg.norm(circuit(CQ(i),None)-controlled(q)),1e-12)
    def test_physical_provider_V_order(self):
        # Tests actual specified order via equality above and confirms inverse uses full circuit.
        self.assertGreater(len(V(2)),len(V(1)))
        self.assertEqual(simplify(V(2)+adjoint(V(2))),[])
    def test_odd_controlled_Pauli(self):
        self.assertLess(np.linalg.norm(circuit(cpauli('XYZ'),None)-controlled(event_operator({'pauli':'XYZ','rotation_sign':0,'phase_i_power':0}))),1e-12)
    def test_direct_off_angle(self):
        e={'word':(1,0),'child':2,'ratio':F(1,3),'phase_i_power':2}
        self.assertLess(circuit_error(e,native_ir(e),None),1e-12)
    def test_helper_off_angle(self):
        e={'word':(2,1),'child':1,'ratio':F(1,3),'phase_i_power':2}
        self.assertLess(circuit_error(e,native_ir(e,True),None,True),1e-12)
    def test_phase_corruption_detected(self):
        e={'word':(),'child':0,'ratio':F(1,3),'phase_i_power':2}
        self.assertGreater(circuit_error(e,native_ir(e)[1:],None),1)
    def test_cts_offdomain_mean(self):
        p=(F(1,7),F(2,7),F(4,7));x=F(1,3);events,cert=cts_events(p,x)
        from trottertracks.algorithm_codesign.g9_matrix import target
        mean=sum(float(e['coefficient'])*event_operator(e) for e in events)
        self.assertLess(np.linalg.norm(mean-target(p,x)),1e-12)
        self.assertLess(F(cert['coefficient_mean_error_upper']),8*F(1,10**12))
    def test_cts_phase_native_offdomain(self):
        for e in cts_events((F(2,9),F(3,9),F(4,9)),F(1,3))[0]:self.assertLess(circuit_error(e,native_ir(e),None),1e-12)
    def test_common_simplification_inverse(self):
        self.assertEqual(simplify([('S',0),('s',0),('CX',0,1),('CX',0,1)]),[])
        self.assertEqual(simplify([('R',0,'1/3',1),('R',0,'1/3',-1)]),[])
        self.assertEqual(len(simplify([('R',0,'1/3',1)]*2)),2)
    def test_zero_T_real_events(self):
        e={'pauli':'XYZ','rotation_sign':0,'phase_i_power':2}
        self.assertEqual(cost(native_ir(e),{})['T'],0)
    def test_confidence_new_row_count(self):
        from trottertracks.algorithm_codesign.g9_comparison import plan
        g=P5Closed((F(2,9),F(7,9)),F(1,3));b=plan(g)
        self.assertEqual(22*b['alpha_axis']+11*b['resource_failure_per_row'],F(1,20))
        self.assertGreater(b['remaining'],0)
    def test_no_new_materiality_threshold(self):
        from trottertracks.algorithm_codesign.g9_comparison import plan
        b=plan(P5Closed((F(2,9),F(7,9)),F(1,3)))
        self.assertNotIn('GO',b)
        self.assertLessEqual(b['accepted_call_cap_two_axes'],b['hard_attempt_cap_two_axes'])
    def test_runner_import_has_no_execution(self):
        import importlib.util
        from pathlib import Path
        p=Path(__file__).resolve().parents[3]/'scripts/tracks/algorithm_codesign/g9_p5_matched_native.py'
        spec=importlib.util.spec_from_file_location('future_g9_runner',p)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        self.assertTrue(callable(mod.execute))


if __name__=='__main__':unittest.main()
