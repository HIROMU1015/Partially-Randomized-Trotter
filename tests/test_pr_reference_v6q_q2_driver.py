"""Q2 driver artificial bookkeeping tests; no scientific builder or input."""
from fractions import Fraction as F
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/run_pr_reference_v6q_q2.py'
spec=importlib.util.spec_from_file_location('q2_driver_tested',path)
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


def inst(name,params=(),wires=()):
    return NS(operation=NS(name=name,params=params),qubits=wires,clbits=())


class DriverTests(unittest.TestCase):
    def test_event_schemas_preserve_sign_and_product_order(self):
        a=dict(order=2,rotation='a',products=['c','b'],angle=-.25,
               phase=[-1.,0.],probability=.125,normalization=1.25)
        b=dict(taylor_order=2,rotation_component_id='a',product_component_ids=['c','b'],
               rotation_angle=-.25,phase=dict(real=-1.,imag=0.),event_probability=.125,event_normalization=1.25)
        self.assertEqual(d.saved_event(a),d.saved_event(b))
        self.assertEqual(d.saved_event(b)['products'],['c','b'])

    def test_common_grid_oracle_handles_large_cancellation(self):
        qc=NS(global_phase=2.**60,data=[inst('rz',(-2.,)),inst('rzz',(2.**61,))])
        self.assertEqual(d.independent_sum(qc),F(1))

    def test_rounding_checker_requires_even_midpoint(self):
        tie=F(1)+F(2)**-53
        self.assertTrue(d.rounding_evidence(tie,1.)['passed'])
        with self.assertRaises(ValueError):
            d.rounding_evidence(tie,math.nextafter(1.,math.inf))

    def test_core_shape_accepts_only_the_relative_P_change(self):
        def core(phase):
            return NS(num_qubits=17,global_phase=0.,find_bit=lambda x:NS(index=x),
                      data=[*[inst('cswap',(),(16,k%8,k%8+8)) for k in range(16)],inst('p',(phase,),(16,))])
        a,b=core(.5),core(.75)
        self.assertTrue(d.core_difference(a,b,.75)['only_ancilla_P_may_differ'])
        b.data[0]=inst('x',(),(0,))
        with self.assertRaises(ValueError):d.core_difference(a,b,.75)

    def test_oracle_does_not_coerce_decimal_parameters(self):
        with self.assertRaises(TypeError):d.rational('0.25')


if __name__=='__main__':unittest.main()
