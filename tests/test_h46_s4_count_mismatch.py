"""Regression evidence for the investigation; no molecular inputs or Qiskit."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h46_s4_audit", ROOT/"scripts/audit_h46_s4_count_mismatch.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class S4MismatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = a.functions()

    def test_negative_coefficient_one_ulp(self):
        native = self.f.weights.yoshida_4th_list()[0]
        w = 1/(2-2**(1/3))
        model = -2**(1/3)*w
        self.assertEqual(native.hex(), "-0x1.b3d16dd72c671p+0")
        self.assertEqual(model.hex(), "-0x1.b3d16dd72c672p+0")
        self.assertEqual(model, math.nextafter(native, -math.inf))

    def test_registered_schedules_differ_only_negative_times(self):
        for q in (1,8):
            model = self.f.m.pf_schedule(3,.8,q,"4th","symmetric_directional")
            native = self.f.schedule.deterministic_control_plan(3,.8,q,"4th")
            for x,y in zip(model,native,strict=True):
                self.assertEqual((x.block,x.mode), (y.term,y.mode.lower()))
                self.assertEqual(x.time == y.time, x.time > 0.)

    def test_second_order_schedules_match(self):
        for q in (1,3,8):
            model = self.f.m.pf_schedule(3,.8,q,"2nd","symmetric_directional")
            native = self.f.schedule.deterministic_control_plan(3,.8,q,"2nd")
            self.assertEqual([(x.block,x.mode,x.time.hex()) for x in model],
                             [(x.term,x.mode.lower(),x.time.hex()) for x in native])

    def test_positive_signature_reproduced_artificially(self):
        r = a.artificial_pair(self.f,(1.,2.,-3.))
        self.assertEqual(r["delta_native_minus_model"],dict(raw=4,rz=2,cx=2,sx=0,x=0))

    def test_negative_signature_reproduced_artificially(self):
        r = a.artificial_pair(self.f,(1.,5.,-6.),q=8)
        self.assertEqual(r["delta_native_minus_model"],dict(raw=-32,rz=-16,cx=-16,sx=0,x=0))

    def test_q_repetition_scales_signature(self):
        for eta in ((1.,2.,-3.),(1.,5.,-6.)):
            one = a.artificial_pair(self.f,eta)["delta_native_minus_model"]
            eight = a.artificial_pair(self.f,eta,q=8)["delta_native_minus_model"]
            self.assertEqual(eight,{k:8*v for k,v in one.items()})

    def test_time_alignment_removes_both_artificial_differences(self):
        for eta in ((1.,2.,-3.),(1.,5.,-6.)):
            for q in (1,8):
                r = a.artificial_pair(self.f,eta,q=q,align_time=True)
                self.assertEqual(set(r["delta_native_minus_model"].values()),{0})

    def test_zero_pruning_not_gate_count_rounding(self):
        r = a.artificial_pair(self.f,(1.,2.,-3.))
        affected = [s for s in r["stages"] if s["model_rz_indices"]!=s["native_rz_indices"]]
        self.assertEqual([s["index"] for s in affected],[2,3])
        self.assertEqual(affected[0]["model_time_hex"],"-0x1.5ca78b1289ec2p-1")
        self.assertEqual(affected[0]["native_time_hex"],"-0x1.5ca78b1289ec1p-1")
        self.assertTrue(any(abs(float.fromhex(t))<1e-14 for t in affected[0]["native_rz_angles_hex"]))
        self.assertEqual(r["phase_nonzero_counts"]["model"],r["phase_nonzero_counts"]["native"])

    def test_shared_primitive_append_semantics(self):
        m,ops=self.f.m,self.f.ops
        for phase in (0.,-0.,.3):
            pp = ops.DiagonalEvolutionPrimitives(phase,((0,0.),(1,-.2)),((0,1,0.),(1,2,.4)))
            for mode in ("UNCONTROLLED","DIRECTIONAL"):
                native = a.append_native(self.f,pp,mode,3)
                model = m.diagonal_from_primitives(pp,mode.lower(),3)
                self.assertEqual(native,model)

    def test_single_retained_rz_signature(self):
        m,ops=self.f.m,self.f.ops
        pp = ops.DiagonalEvolutionPrimitives(0.,((0,1e-30),),())
        uncontrolled = m.count(a.append_native(self.f,pp,"UNCONTROLLED",1))
        directional = m.count(a.append_native(self.f,pp,"DIRECTIONAL",1))
        self.assertEqual((uncontrolled+directional).as_dict(),dict(raw=4,rz=2,cx=2,sx=0,x=0))

    def test_zero_primitive_and_nonfinite_guard(self):
        m,ops=self.f.m,self.f.ops
        for zero in (0.,-0.):
            pp = ops.one_body_diagonal_primitives(a.np.asarray([zero]),.2)
            self.assertEqual(pp.rz,())
        with self.assertRaises(ValueError):
            m.diagonal_from_primitives(ops.DiagonalEvolutionPrimitives(0.,((0,float('nan')),),()),"directional",1)

    def test_lowering_is_angle_independent_without_pruning(self):
        m=self.f.m
        for gate,qubits in (("rz",(0,)),("rzz",(0,1)),("xx_plus_yy",(0,1))):
            zero=m.gate_cost(m.Gate(gate,qubits,0.))
            self.assertEqual(zero,m.gate_cost(m.Gate(gate,qubits,1e-30)))
            self.assertEqual(zero,m.gate_cost(m.Gate(gate,qubits,-.7)))

    def test_axis_wrapper_offsets_preserved(self):
        m=self.f.m
        c=m.Circuit((m.Gate("rz",(0,),.3),),.2)
        x=m.count(m.hadamard_wrapper(c,1,"X"))
        y=m.count(m.hadamard_wrapper(c,1,"Y"))
        self.assertEqual(y.rz-x.rz,1)
        self.assertEqual(y.cx,x.cx)
        self.assertEqual(m.hadamard_wrapper(c,1,"Y").global_phase,.2)

    def test_saved_inventory_and_raw_signature(self):
        saved=a.saved_evidence()
        self.assertEqual(len(saved["mismatches"]),8)
        self.assertEqual(len(saved["controls"]),20)
        for row in saved["saved_cost_signatures"]:
            d=row["delta_native_minus_theory"]
            self.assertEqual(d["rz"],d["cx"])
            self.assertEqual(d["raw"],2*d["rz"])
            self.assertEqual((d["sx"],d["x"]),(0,0))
            self.assertTrue(row["zero_unknown_primitives"])
            expected=1 if row["system"]=="H4" else -8
            self.assertEqual(row["inferred_net_cardinality_change"]["retained_rz"],expected)
            self.assertEqual(row["inferred_net_cardinality_change"]["nonzero_rzz"],0)
            self.assertEqual(row["inferred_net_cardinality_change"]["nonzero_phase"],0)

    def test_cardinality_inverse_does_not_conflate_phase_rzz(self):
        # Three different emission mechanisms have different raw/RZ/CX vectors.
        for da,db,dg in ((1,0,0),(0,1,0),(0,0,1),(-3,2,-1)):
            d=dict(raw=4*da+6*db+dg,rz=2*da+2*db+dg,cx=2*da+6*db,sx=0,x=0)
            result=a.infer_diagonal_cardinality(d)
            self.assertEqual((result['retained_rz'],result['nonzero_rzz'],result['nonzero_phase']),(da,db,dg))

    def test_no_qiskit_or_scientific_provider_import(self):
        self.assertFalse(any(k=="qiskit" or k.startswith("qiskit.") for k in sys.modules))
        self.assertNotIn("trottertracks.resource_applicability.ax2b_native_df_v5",sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
