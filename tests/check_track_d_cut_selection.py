"""Invented metadata + negative gates only, not molecular fixtures."""
from dataclasses import replace
import inspect
import unittest
from qiskit import QuantumCircuit
from trottertracks import track_d_cut_selection as s
from trottertracks import track_d_cut_artificial as a


def metadata(runs):
    return [[dict(rotation_angle=.2,application_sequence=[dict(
        basis_id=str(i%r),diagonal_pauli_support=[0],coefficient_sign=(-1)**i)])
             for i in range(16)] for r in runs]


class CutTests(unittest.TestCase):
    features=None

    def test_imbalanced_neighbor_is_selected_before_compile(self):
        d=s.select_cut(metadata([16,16,16,1,1,1]),type(self).features)
        self.assertEqual((d.center,d.selected),(3,2))
        self.assertEqual([x[0] for x in d.candidates],[2,3,4])

    def test_balanced_center_fallback(self):
        d=s.select_cut(metadata([16]*6),type(self).features)
        self.assertEqual(d.selected,3)
        self.assertGreater(d.candidates[0][1],d.candidates[1][1])

    def test_short_path_has_only_internal_candidates(self):
        d=s.select_cut(metadata([1,1]),type(self).features)
        self.assertEqual((d.center,d.selected,len(d.candidates)),(1,1,1))
        with self.assertRaises(ValueError):s.select_cut(metadata([1]),type(self).features)

    def test_bad_descriptor_and_zero_application_are_rejected(self):
        for key,value in (("basis_id",None),("coefficient_sign",0),("diagonal_pauli_support",[0,0])):
            rows=metadata([1]*4);rows[0][0]["application_sequence"][0][key]=value
            with self.assertRaises(ValueError):s.select_cut(rows,type(self).features)
        rows=metadata([1]*4);rows[0][0]["rotation_angle"]=float("nan")
        with self.assertRaises(ValueError):s.select_cut(rows,type(self).features)

    def test_compiled_price_fields_are_not_features(self):
        rows=metadata([16,16,16,1,1,1]);expected=s.select_cut(rows,type(self).features)
        for step in rows:
            for e in step:e.update(compiled_depth=-999999,Born=1.,oracle_best_cut=4)
        self.assertEqual(s.select_cut(rows,type(self).features),expected)
        self.assertNotIn("transpile",inspect.getsource(s.select_cut))

    def test_illegal_cut_and_descriptor_mismatch_stop_before_builder(self):
        rows=metadata([1]*4);d=s.select_cut(rows,type(self).features)
        steps=[dict(events=x,sample=i) for i,x in enumerate(rows)]
        for cut in (0,4,-1):
            with self.assertRaisesRegex(ValueError,"CUT_OR_DESCRIPTOR"):
                s.build_selected(None,steps,replace(d,selected=cut))
        rows[0][0]["rotation_angle"]=-.2
        with self.assertRaisesRegex(ValueError,"CUT_OR_DESCRIPTOR"):s.build_selected(None,steps,d)

    def test_unsupported_or_nonvacuum_half_is_not_eligible(self):
        c=QuantumCircuit(2);c.h(0)
        with self.assertRaisesRegex(ValueError,"UNSUPPORTED"):a.eligible_half(c,"negative_H")
        c=QuantumCircuit(2);c.x(0)
        with self.assertRaises(ValueError):a.eligible_half(c,"negative_nonvacuum")

    def test_qubit_bit_readout_and_negative_phase(self):
        A=QuantumCircuit(2);A.rz(.31,0)
        B=QuantumCircuit(2);B.rz(-.17,1)
        V=a.Operator(A).data@a.Operator(B).data;chi=(-.31+.17)/2
        correct=a.toy.c3.base.reference_gadget(A,B,chi)
        wrong=a.toy.c3.base.reference_gadget(A,B,-chi)
        for axis in ("X","Y"):
            w=a.toy.c3.base.artificial.hadamard_wrapper(correct,axis)
            self.assertEqual(w.find_bit(w.data[-1].qubits[0]).index,4)
            self.assertEqual(w.find_bit(w.data[-1].clbits[0]).index,0)
            a.toy.check(a.toy.c3.base.artificial.induced_observable(w),
                       a.toy.c3.base.artificial.expected_observable(V,axis),"bit0+1_bit1-1",observable=True)
        w=a.toy.c3.base.artificial.hadamard_wrapper(wrong,"Y")
        self.assertGreater(float(a.np.max(abs(a.toy.c3.base.artificial.induced_observable(w)-
            a.toy.c3.base.artificial.expected_observable(V,"Y")))),.01)
