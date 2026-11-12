"""Synthetic local words and immutable public metadata; no molecular load."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import h4_pr_physical_closure as d
from trottertracks.h4_df10_price_v1 import Accounting


class Tests(unittest.TestCase):
    def test_law_descriptor_uses_original_integer_API_without_sampling(self):
        def make(tail,**kw):
            self.assertEqual(kw['seed'],0);self.assertIs(type(kw['seed']),int)
            self.assertEqual(kw['finite_taylor_order'],2)
            return NS(dimensionless_step_time=tail.lambda_r*(kw['evolution_time']/kw['rte_steps'])),NS(orders=(0,2))
        context=NS(rte=NS(make_rte_config=make),tolerance=lambda tau,K:1e-12)
        for R in (2,128):
            tau,_,_=d.make_law(context,NS(lambda_r=1.25),.1,R)
            self.assertEqual(tau,1.25*(.1/R))

    def test_exact_binary64_suffix_and_reset_not_modulo(self):
        def rz(x):return dict(name='rz',wires=[0],angle=x)
        self.assertEqual(d.wire_effect([rz(.5),rz(-.5)],0)['exit_z'],0.)
        self.assertEqual(d.wire_effect([rz(1.),rz(2**-54),rz(-1.)],0)['exit_z'],0.)
        e=d.wire_effect([rz(.7),dict(name='sx',wires=[0],angle=None),rz(.25)],0)
        self.assertTrue(e['reset']);self.assertEqual(e['exit_z'],.25)
        self.assertEqual(e['leading_z'],[.7]);self.assertTrue(d.wire_effect([rz(6.283185307179586)],0)['z_only'])

    def test_original_native_expansion_and_product_rotation_control_flush(self):
        T=d.tracer(Accounting)
        for support in ((0,),(1,6)):
            for role in ('product','rotation'):
                word=d.application_word(T,support,role,.17)
                self.assertTrue(d.wire_effect(word,8)['reset'])
                self.assertIsNone(d.wire_effect(word,8)['exit_z'])
        with self.assertRaises(RuntimeError):T().finish()

    def test_reset_exit_matches_original_pending_Z_for_all_tested_inputs(self):
        word=[dict(name='rz',wires=[0],angle=.125),dict(name='sx',wires=[0],angle=None),
              dict(name='rz',wires=[0],angle=.25),dict(name='rz',wires=[0],angle=-.125)]
        effect=d.wire_effect(word,0)
        for initial in (None,0.,.3,-.125):
            a=Accounting();a.pending_z[0]=initial
            for op in word:a._native(op['name'],op['wires'],op['angle'])
            self.assertEqual(a.pending_z[0],effect['exit_z'])

    def test_basis_delayed_inverse_literal_equality_and_wire_order(self):
        T=d.tracer(Accounting);ops=[dict(name='p',qubits=[0],parameters=['0.3'])]
        t=T();t.basis(ops,True,'a');t.basis(ops,False,'b');self.assertEqual(t.word,[])
        self.assertIsNone(t.pending_basis)
        x=d.basis_word(T,ops,True);self.assertEqual(x[0]['angle'],-.3)
        self.assertTrue(d.wire_effect(x,0)['z_only']);self.assertTrue(d.wire_effect(x,1)['untouched'])

    def test_tail_phase_is_not_chunked_or_reassociated(self):
        # Original tail has one local active/phase accumulator per legal tail.
        import inspect
        src=inspect.getsource(Accounting.tail)
        self.assertIn('phase +=',src);self.assertIn('for event in events:',src)
        self.assertIn('if phase != 0:',src)

    def test_correlated_wire_marginals_reuse_MeanKernel_without_independence(self):
        import itertools
        from trottertracks.mean_compiled_price import MeanKernel
        words=[[dict(name='x',wires=[0,1],angle=None),dict(name='rz',wires=[0],angle=.25)],
               [dict(name='x',wires=[0,1],angle=None),dict(name='rz',wires=[1],angle=.25)]]
        # The x on two wires is only a transport shorthand for two literal Xs.
        words=[[dict(name='x',wires=[0],angle=None),dict(name='x',wires=[1],angle=None),w[-1]] for w in words]
        means=[]
        for q in (0,1):
            kernel=MeanKernel.from_events((None,.25),(0,1),(.4,.6),lambda z,i:d.projected_step(z,words[i],q))
            means.append(kernel.repeat(3).expected([1.,0.],[0.,1.])[0])
        total=0.
        for path in itertools.product((0,1),repeat=3):
            a=Accounting();weight=1.
            for i in path:
                weight*= (.4,.6)[i]
                for op in words[i]:a._native(op['name'],op['wires'],op['angle'])
            total+=weight*a.finish()['structural_rz']
        self.assertAlmostEqual(sum(means),total)

    def test_positive_phase_barrier_keeps_literal_two_pi_distinct_from_zero(self):
        import math
        word=[dict(name='cx',wires=[8,0],angle=None),dict(name='rz',wires=[8],angle=.3)]
        for phase in (math.pi,2*math.pi,3*math.pi):
            self.assertEqual(d.projected_step(phase,word,8),(.3,1))
        self.assertEqual(d.projected_step(0.,word,8),(.3,0))

    def test_formal_B_configuration_matches_prior_C_requests(self):
        plan,match=d.check_publication(ROOT)
        self.assertEqual(match['configuration_id'],plan['B_V2_configuration_id'])
        self.assertEqual([r['m'] for r in plan['focus_requests']],[0,6])

    def test_import_metadata_check_do_not_run_science(self):
        with patch.object(d,'diagnostic',side_effect=AssertionError('No science')):
            d.check_publication(ROOT)


if __name__=='__main__':unittest.main(verbosity=2)
