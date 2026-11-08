"""Artificial tests only: no saved scientific guard/T evaluation."""
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import audit_pr_reference_inverse_d_static as a
import diagnose_pr_reference_inverse_d as n


class InverseDTests(unittest.TestCase):
    def test_static_pool_preserves_unordered_guard_T_strings(self):
        entry=dict(target_literal="artificial",roles=[],current_word=dict(sequence_sha256="current"),exact_literal_records=[])
        frontier=dict(nondominated_ids=["fixed_current","other"],offers=[
            dict(id="fixed_current",sequence_sha256="current",guard="not-a-number"),
            dict(id="other",sequence_sha256="other",guard="also-not-a-number")])
        r=a.pool_metadata(entry,frontier,dict(global_W="saved-only"))
        self.assertTrue(r["current_retained_saved"])
        self.assertEqual(r["frontier_distinct_sequence_count"],2)
        self.assertEqual(r["smaller_guard_or_higher_T_direction"],"NOT_NEWLY_COMPARED")

    def test_native_metadata_keeps_sign_wire_order_and_missing_phase(self):
        rec=dict(name="basis_inverse:x",n=2,lowering=[
            ["rz",[1],[0.5]],["cx",[0,1],[]],["rz",[0],[-0.5]]])
        r=a.native_binding("b","saved","member",dict(key="c"),rec,"frame_inverse")
        self.assertEqual([x["native_position"] for x in r["ordered_RZs"]],[0,2])
        self.assertEqual([x["wires"] for x in r["ordered_RZs"]],[[1],[0]])
        self.assertEqual(r["physical_key_multiplicities"][(0.5).hex()],2)
        self.assertEqual(r["native_global_phase"],"NOT_SAVED_NOT_ZERO")

    def test_SOURCE_phase_is_not_native_phase(self):
        rec=dict(name="block:0:half",inverse_global_phase=0.25,
            native=dict(n=1,lowering=[["rz",[0],[0.5]]]))
        r=a.native_binding("inv","saved","member",dict(context=[False,False]),rec,"D")
        self.assertEqual(r["SOURCE_global_phase_fields_quoted"],dict(inverse_global_phase=0.25))
        self.assertEqual(r["native_global_phase"],"NOT_SAVED_NOT_ZERO")

    def test_reused_rules_keep_current_on_nondominated_NT_case(self):
        f=n.reuse()
        self.assertNotIn("main",f)
        pool=dict(offers=[dict(id="fixed_current",guard="1",T=10),
            dict(id="precise",guard="1/10",T=20),dict(id="cheap",guard="2",T=5)],
            nondominated_ids=["fixed_current","precise","cheap"])
        self.assertEqual([f["choose_offers"](pool,p)["id"] for p in n.POLICIES],["precise","cheap","fixed_current"])

    def test_D_coefficients_match_old_arm_expressions(self):
        f=n.reuse();control,legacy,arms,_=f["load_arithmetic"]()
        ids=("artificial",);alphabet={arm:ids for arm in arms};required={}
        for arm in arms:required.update(control["required_templates"](arm,ids))
        values={m:{name:F(i+1,7) for i,name in enumerate(names)} for m,names in required.items()}
        for arm in arms:
            actual=arms[arm](4,2,values["b"],values["ref"],values["inv"],alphabet)["D"]
            expected=sum((c*values[m][t] for (m,t),c in n.d_multiplicities(4,arm).items()),F(0))
            self.assertEqual(actual,expected)

    def test_shared_key_is_propagated_to_inverse_and_D(self):
        f=n.reuse();_,legacy,_,_=f["load_arithmetic"]()
        key=(0.5).hex();batches={"b":[dict(key="c",records=[
            dict(name="basis_inverse:x",lowering=[["rz",[0],[0.5]]]),
            dict(name="left_all",lowering=[["rz",[1],[-0.5]]])])]}
        rows=f["context_details"](batches,{key:F(1,8)},{key:3},legacy["weighted_template_max"],[key])
        self.assertEqual([r["logical_guard"] for r in rows],[F(1,8),F(1,8)])
        self.assertEqual([r["local_template_T_from_saved_prices"] for r in rows],[3,3])

    def test_D_attribution_is_separate_max_not_reachable_joint_path(self):
        weights={"b":{name:F(2) for m,name in n.d_multiplicities(4,"Bprime")}}
        rows=[dict(mapping="b",template=name,context=c,logical_guard=F(value),
            target_key_contributions={"artificial":dict(guard=F(value),multiplicity=1,local_T=3)})
            for m,name in n.d_multiplicities(4,"Bprime") for c,value in (("b",2),("a",2))]
        r=n.d_attribution("Bprime",4,weights,rows)
        self.assertEqual(r["D_upper"],sum(c*2 for c in n.d_multiplicities(4,"Bprime").values()))
        self.assertTrue(all(x["selected_context"]=="a" for x in r["terms"]))
        rows[0]["logical_guard"]=None
        with self.assertRaises(ValueError):n.d_attribution("Bprime",4,weights,rows)


if __name__=="__main__":unittest.main()
