"""Synthetic control/phase and fixed-order checks; no research inputs."""
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import json
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_track_d_c3_q4_q8 as runner


class C3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runner.c2.old().bootstrap("/tmp/track-d-paired32-20261106/origin")
        from trottertracks import track_d_c3_whole_path
        cls.m=track_d_c3_whole_path

    def test_contract_loads_from_execution_tree_without_origin_contract(self):
        with tempfile.TemporaryDirectory(prefix="track-d-c3-contract-") as directory:
            root=Path(directory)/"execution"; origin=Path(directory)/"original_SOURCE";origin.mkdir()
            relative=runner.c2.old().CONTRACT
            path=root/relative;path.parent.mkdir(parents=True)
            packet={"input":"saved-input.npz","candidate":"saved-endpoint.npz"}
            path.write_text(json.dumps(packet))
            self.assertFalse((origin/relative).exists())
            self.assertEqual(runner.load_contract(root),packet)
        self.assertEqual(runner.load_contract(),json.loads((ROOT/relative).read_text()))

    def test_fixtures_and_half_step_boundary_control_mode(self):
        self.assertEqual(self.m.FIXTURES,{4:tuple(range(4)),8:tuple(range(8))})
        self.assertEqual(self.m.CUTS,{4:2,8:4})
        recipe=self.m.head_recipe("boundary")
        self.assertEqual([x[0] for x in recipe],[2,1,0,1,2])
        self.assertEqual(recipe[2],(0,Fraction(1,10),"ordinary"))
        self.assertEqual([x[2] for x in recipe],["directional","directional","ordinary","uncontrolled","uncontrolled"])

    def test_mixed_boundary_preserves_both_control_branches_and_scalar(self):
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        from trotterlib.df_trotter.ops import DiagonalEvolutionPrimitives
        diagonal=DiagonalEvolutionPrimitives(.19,((0,.31),),())
        model=SimpleNamespace(modes=1,blocks=[SimpleNamespace(rotation=((1.,),),diagonal=diagonal)])
        basis=QuantumCircuit(1);basis.ry(.43,0);basis.global_phase=.11
        builder=SimpleNamespace(relative_fusion=True,_basis=lambda model,block:basis)
        context=SimpleNamespace(builder=builder,arm=SimpleNamespace(model=model))
        half=Fraction(1,20)
        directional=self.m.mixed_head(context,((0,half,"directional"),))
        forward=self.m.mixed_head(context,((0,half,"uncontrolled"),))
        fused=self.m.mixed_head(context,((0,2*half,"ordinary"),))
        explicit=Operator(directional.compose(forward)).data
        np.testing.assert_allclose(Operator(fused).data,explicit,atol=1e-14,rtol=0)
        np.testing.assert_allclose(explicit[:2,:2],np.eye(2),atol=1e-14,rtol=0)
        unctrl=self.m.mixed_head(context,((0,2*half,"uncontrolled"),))
        np.testing.assert_allclose(explicit[2:,2:],Operator(unctrl).data[:2,:2],atol=1e-14,rtol=0)
        wrong=Operator(self.m.mixed_head(context,((0,2*half,"directional"),))).data
        self.assertGreater(float(np.max(abs(wrong-explicit))),1e-4)

    def test_comparison_keeps_saved_evidence_separate(self):
        rows={q:{arm+"_"+axis:{"metrics":{k:10*q+(1 if arm=="reference" else 0) for k in runner.c2.METRICS}}
                 for arm in runner.c2.ARMS for axis in runner.c2.AXES} for q in runner.QS}
        saved={"q2_paired":["saved bytes only"]}
        value=runner.comparisons(rows,saved)
        self.assertEqual(value["saved_q1_q2"],saved)
        self.assertEqual(set(value["current"]),{"4","8"})
        self.assertEqual(value["current"]["4"][0]["delta"]["cx_count"],1)


if __name__=="__main__":unittest.main()
