"""Artificial registry/interval/accounting tests; no real H4 input reads."""
import copy
from fractions import Fraction as Q
import importlib
from pathlib import Path
import sys
import types
import unittest

pkg=types.ModuleType("b0_comparison_test");pkg.__path__=[str(Path(__file__).resolve().parents[1]/"src/trotterlib")];sys.modules[pkg.__name__]=pkg
F=importlib.import_module(pkg.__name__+".h4_pauli_b0_comparison")


def target(bias="0",u="0"):
    return dict(axes={a:dict(signed_bias_exact=bias,u_enc_exact=u) for a in F.AXES})


class ComparisonTests(unittest.TestCase):
    def test_candidate_order_and_exact_time(self):
        base=dict(n_qubits=2,qubit_order="qiskit_little_endian",state=[[1,0]]+[[0,0]]*3,hamiltonian=dict(constant=.1,terms=[["ZI",.7],["IZ",-.4],["XX",.2]]))
        old=copy.deepcopy(base)
        for q in (1,2,4,8):
            m=F.candidate_model(base,2,q)
            self.assertEqual([b["label"] for b in m["cases"]["PR"]["blocks"]],["ZI","IZ"])
            self.assertEqual(Q(m["step_time"])*q,Q(.8))
        self.assertEqual(base,old)
        self.assertTrue(F.commuting_prefix(base,2));self.assertFalse(F.commuting_prefix(base,3))
        with self.assertRaises(ValueError):F.candidate_model(base,4,1)

    def test_saved_full_target_is_whole_nominal_interval(self):
        C=dict(bits=64,nominal_law=dict(PF=dict(signed_bias=dict(real=dict(lower="1",upper="3"),imag=dict(lower="-2",upper="0")))))
        t=F.reuse_full_pf_target(C,"5")
        self.assertEqual(t["axes"]["cosine"]["signed_bias_exact"],"2/5")
        self.assertEqual(t["axes"]["cosine"]["u_enc_exact"],"1/5")
        self.assertEqual(t["state_actions"],0)

    def test_null_is_not_an_infinite_or_zero_price(self):
        bad=F.account(target("1"),dict(cosine=None,sine=None),"0.05")
        self.assertIsNone(bad["G_exact"]);self.assertIsNone(bad["N_total"])
        self.assertIsNone(F.finite_minimum([dict(bad,candidate_id="bad")]))
        self.assertEqual(F.joint_status(target("1"),"0.05"),"BIAS_INELIGIBLE")

    def test_own_guard_and_family_minimum(self):
        rows=[]
        for name,cost in (("a",10),("b",20)):
            row=F.account(target("0","1/1000000000000"),dict(cosine=cost,sine=cost),"0.05")
            self.assertEqual(Q(row["G_exact"]),row["N_total"]*cost)
            self.assertIsNone(row["actual_G"])
            rows.append(dict(row,candidate_id=name))
        best=F.finite_minimum(rows)
        self.assertEqual(best["candidate_ids"],["a"])
        comp=F.compare_ranges(best["independent_guard_range_exact"],[str(Q(rows[1]["G_exact"])*10),str(Q(rows[1]["G_exact"])*11)])
        self.assertEqual(comp["independent_guard_order"],"DETERMINISTIC_LOWER_ALL_GUARDS")

    def test_ties_and_overlapping_guards(self):
        r=F.account(target(),dict(cosine=1,sine=1),"0.05")
        b=F.finite_minimum([dict(r,candidate_id="a"),dict(r,candidate_id="b")])
        self.assertEqual(b["candidate_ids"],["a","b"])
        self.assertEqual(F.compare_ranges(["10","12"],["11","13"])["independent_guard_order"],"GUARD_RANGES_OVERLAP")


if __name__=="__main__":unittest.main()
