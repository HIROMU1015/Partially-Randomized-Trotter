"""Artificial operands only. No saved guard/T selection or numeric main."""
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/(name+".py"))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


n=module("diagnose_pr_reference_frame_impact")
s=module("audit_pr_reference_frame_impact_static")


def offer(i,g,t):return dict(id=i,guard=str(g),T=t,sequence_sha256="artificial:"+i)


class PreparationTests(unittest.TestCase):
    def test_three_finite_rules_preserve_guard_T_tradeoff(self):
        f=dict(offers=[offer("fixed_current",1,10),offer("guard",F(1,10),12),
            offer("cheap",F(4,5),3),offer("no_increase",F(1,5),9)],
            nondominated_ids=["guard","cheap","no_increase"])
        self.assertEqual([n.choose_offers(f,k)["id"] for k in n.POLICIES],["guard","cheap","no_increase"])

    def test_use_only_saved_frontier_and_stable_ID_tie(self):
        f=dict(offers=[offer("fixed_current",1,10),offer("z",F(1,4),7),
            offer("a",F(1,4),7),offer("excluded",0,0)],nondominated_ids=["z","a"])
        self.assertEqual(n.choose_offers(f,"G")["id"],"a")
        self.assertNotEqual(n.choose_offers(f,"T")["id"],"excluded")

    def test_no_T_increase_rule_has_explicit_current_fallback(self):
        f=dict(offers=[offer("fixed_current",1,2),offer("expensive",F(1,10),9)],nondominated_ids=["expensive"])
        self.assertEqual(n.choose_offers(f,"NT")["id"],"fixed_current")

    def test_missing_offer_or_negative_guard_rejected(self):
        for f in [dict(offers=[offer("fixed_current",1,2)],nondominated_ids=["absent"]),
                  dict(offers=[offer("fixed_current",1,2),offer("bad",-1,1)],nondominated_ids=["bad"])]:
            with self.assertRaises(ValueError):n.choose_offers(f,"G")

    def test_legacy_SOURCE_only_reuse_and_missing_guard_no_zero_fill(self):
        control,legacy,arms,evidence=n.load_arithmetic()
        self.assertNotIn("main",legacy);self.assertNotIn("pareto_ids",legacy)
        w=legacy["weighted_template_max"]
        self.assertIsNone(w([{"missing":1}],{}))
        self.assertEqual(w([{"x":2},{"x":3}],{"x":F(1,4)}),F(3,4))
        batch={"b":[dict(key="artificial",records=[dict(name="basis:x",lowering=[
            ["rz",[0],[0.5]],["rz",[1],[-0.5]]])])]}
        rows=n.context_details(batch,{}, {}, w,[(0.5).hex()])
        self.assertIsNone(rows[0]["logical_guard"])
        self.assertIsNone(rows[0]["local_template_T_from_saved_prices"])
        self.assertEqual(rows[0]["target_key_contributions"][(0.5).hex()]["multiplicity"],2)

    def test_forward_inverse_keep_separate_context_maxima(self):
        weights={"b":{"basis:x":F(4),"basis_inverse:x":F(5)}}
        details=[dict(mapping="b",template=t,context=c,logical_guard=F(g))
            for t,c,g in [("basis:x","a",4),("basis:x","b",1),
                          ("basis_inverse:x","a",1),("basis_inverse:x","b",5)]]
        r=n.witnesses(weights,details,"b",["x"])
        self.assertEqual(r["pair_maximum"],9)
        self.assertEqual(r["per_template_context_witnesses"]["basis:x"]["tied_contexts"],["a"])
        self.assertEqual(r["per_template_context_witnesses"]["basis_inverse:x"]["tied_contexts"],["b"])

    def test_argmax_ties_and_missing_template_retained(self):
        weights={"b":{"basis:x":F(2),"basis_inverse:x":F(1),"basis:y":F(1),"basis_inverse:y":F(2)}}
        details=[dict(mapping="b",template=t,context="a",logical_guard=g) for t,g in weights["b"].items()]
        self.assertEqual(n.witnesses(weights,details,"b",["x","y"])["tied_basis_ids"],["x","y"])
        self.assertIsNone(n.witnesses(weights,details,"b",["missing"]))

    def test_static_crosswalk_quotes_nonnumeric_fields_without_parsing(self):
        hit=dict(arm="Bprime",archive_member="artificial",context="c",template="basis:x",
            role="SOURCE_RTE_frame",native_position=1,wires=[0],signed_literal=(0.5).hex())
        a=dict(source_RTE_basis_ids=["x","y"],candidates_43=[dict(literal=(0.5).hex(),
            saved_template_hits=[hit],nondominated_ids_saved=["saved"],candidate_metadata=[dict(guard="not-a-number",T="quoted-only")])])
        b=dict(template_weights={"b":{"basis:x":"raw-value-only"}},rows=[])
        m=dict(entries=[dict(target_literal=(0.5).hex(),current_word=dict(guard="quoted"),roles=[])])
        r=s.crosswalk(a,b,m)
        self.assertEqual(r["unaffected_SOURCE_basis_ids"],["y"])
        self.assertFalse(r["guard_T_numeric_comparison"])
        self.assertEqual(r["candidates_43"][0]["contexts"][0]["S1_template_max_quoted"],"raw-value-only")


if __name__=="__main__":unittest.main()
