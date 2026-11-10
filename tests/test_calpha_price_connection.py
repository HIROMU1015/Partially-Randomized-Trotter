"""Synthetic fixed-K0 pricing connection; no molecular data or prices."""
import unittest
from dataclasses import replace
from fractions import Fraction
from test_b2_rpe_connection import case,fake_compile
from trotterlib.b2_p1_compiled_cost import price_explicit_trajectory
from trotterlib.o1_m1_resource_bridge import MatchedNoPrepCostProvider
from trotterlib.rte import enumerate_rte_events


class PriceConnectionTests(unittest.TestCase):
    def test_two_measured_axes_and_no_expectation_claim(self):
        _,r=case(r=4,K=0)
        events=enumerate_rte_events(r.tail.components,r.distribution)
        calls=[]
        def capture(circuit,settings,**kwargs):
            calls.append(circuit)
            return fake_compile(circuit,settings,**kwargs)
        result=price_explicit_trajectory(r,MatchedNoPrepCostProvider(r.context.compiler,compile_function=capture),[events[0]]*4)
        self.assertEqual(len(calls),2)
        self.assertIsNone(result['expected_trajectory_cost'])
        self.assertIsNone(result['total_rpe_cost'])
        self.assertFalse(result['state_preparation_included'])

    def test_noncanonical_k0_angle_rejected_before_compile(self):
        _,r=case(r=4,K=0)
        event=enumerate_rte_events(r.tail.components,r.distribution)[0]
        bad=replace(event,rotation_angle=event.rotation_angle+.01)
        with self.assertRaises(ValueError):
            price_explicit_trajectory(r,MatchedNoPrepCostProvider(r.context.compiler,compile_function=fake_compile),[bad]*4)

    def test_exact_basis_reuse_expectation_matches_all_two_event_circuits(self):
        from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
        from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
        _,r=case(r=2,K=0,L_D=1)
        pool=enumerate_rte_events(r.tail.components,r.distribution)
        events=[pool[0],next(x for x in pool if x.basis_id!=pool[0].basis_id)]
        probabilities=[Fraction(3,10),Fraction(7,10)]
        prep=r.preparation.rte_preparation
        lengths=[len(prep.basis_registry.operations(e.basis_id)) for e in events]
        builder=QiskitDFRTEEventCircuitBuilder(basis_registry=prep.basis_registry)
        measured=Fraction(0)
        for i in range(2):
            for j in range(2):
                request=DFRTEEventSequenceCircuitRequest(events=(events[i],events[j]),
                    component_specs=prep.component_specs,controlled=True,ancilla_qubit=2,
                    tail_id=r.tail.tail_id,tail_hash=r.tail.tail_hash,occurrence_rte_steps=2)
                measured+=probabilities[i]*probabilities[j]*len(builder.build_sequence(request).circuit.data)
        expected=2+2*sum((length*(p+p*(1-p)) for length,p in zip(lengths,probabilities)),Fraction(0))
        self.assertEqual(measured,expected)


if __name__=='__main__':unittest.main()
