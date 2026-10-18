"""Synthetic multinomial and two-qubit rare-event native regression tests."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h6_dba_contract as c
import h6_dba_sampling as sampling


class Counts(unittest.TestCase):
    def test_multinomial_normalization_against_enumeration(self):
        import itertools
        for stratum in ['s2_multi','s4']:
            dist=SimpleNamespace(orders=(0,2,4),order_probabilities=(.6,.3,.1))
            law=sampling.count_law(4,dist,stratum)
            expected={}
            for orders in itertools.product([0,2,4],repeat=4):
                if sampling.counts.in_stratum(orders.count(2),orders.count(4),stratum):
                    k=orders.count(2),orders.count(4)
                    expected[k]=expected.get(k,0)+math.prod({0:.6,2:.3,4:.1}[x] for x in orders)
            mass=math.fsum(expected.values())
            self.assertAlmostEqual(math.exp(law.log_stratum_mass),mass,places=14)
            for key,p in zip(law.counts,law.probabilities):self.assertAlmostEqual(p,expected[key]/mass,places=14)

    def test_k2_has_no_fourth_order_support(self):
        d=SimpleNamespace(orders=(0,2),order_probabilities=(.9,.1))
        law=sampling.count_law(16,d,'s2_multi')
        self.assertTrue(all(n2>=2 and n4==0 for n2,n4 in law.counts))
        self.assertGreater(max(n2 for n2,n4 in law.counts),2)
        with self.assertRaises(ValueError):sampling.count_law(16,d,'s4')

    def test_fourth_layer_includes_multiple_events(self):
        d=SimpleNamespace(orders=(0,2,4),order_probabilities=(.6,.3,.1))
        law=sampling.count_law(8,d,'s4')
        self.assertIn((1,2),law.counts)

    def test_unrepresentable_support_rejected(self):
        d=SimpleNamespace(orders=(0,2,4),order_probabilities=(1.,1e-100,1e-200))
        with self.assertRaisesRegex(ValueError,'UNREPRESENTABLE'):
            sampling.count_law(16,d,'s4')


class NativeFixture(unittest.TestCase):
    def test_two_qubit_k2_multi_and_k4_event_oracle(self):
        import numpy as np
        from trotterlib.df_hamiltonian import DFHamiltonian
        from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
        from trottertracks.resource_applicability import h6_matched_port_v1 as m
        from trottertracks.resource_applicability import ax2b_native_df_v5 as native
        from h6_dba_science import event_oracle
        ham=DFHamiltonian(.17,np.array([[.2,.04],[.04,-.3]]),np.array([.4,-.2]),
            (np.array([[.6,.1],[.1,-.2]]),np.array([[.3,.07],[.07,.5]])),{})
        prep=_prepare(ham,1)
        state=np.array([0,1,0,0],dtype=complex)
        for K,stratum in [(2,'s2_multi'),(4,'s4')]:
            cell=c.cell(1,2,2,K)
            steps,receipt=sampling.steps(prep,cell,stratum,1234+K)
            again,receipt2=sampling.steps(prep,cell,stratum,1234+K)
            self.assertEqual(receipt,receipt2)
            self.assertFalse(receipt['quantum_weights_changed'])
            self.assertFalse(receipt['normalization_changed'])
            if K==2:self.assertGreaterEqual(receipt['n2'],2)
            else:self.assertGreaterEqual(receipt['n4'],1)
            evolution=native.partial_native_from_step_requests(steps,control_policy='symmetric_directional',max_instructions=100000)
            low=np.r_[state,np.zeros_like(state)];high=np.r_[np.zeros_like(state),state]
            a=m.simulate_statevector(evolution.circuit,low)
            b=m.simulate_statevector(evolution.circuit,high)
            self.assertLess(np.linalg.norm(a-low),1e-12)
            z=np.vdot(state,b[4:]);independent=event_oracle(steps,m,native,state)
            self.assertLess(abs(z-independent),1e-12)
            for axis,target in [('cosine',z.real),('sine',z.imag)]:
                wrapper=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=False,max_instructions=100000)
                self.assertLess(abs(m.hadamard_expectation(m.simulate_statevector(wrapper,low),2)-target),1e-12)


if __name__=='__main__':unittest.main()
