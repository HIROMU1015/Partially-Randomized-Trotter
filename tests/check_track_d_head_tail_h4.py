"""Artificial eight-mode span/gauge tests; never loads molecular data/P0."""
import importlib.util
import math
import json
from pathlib import Path
import sys
from types import SimpleNamespace as S
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("head_tail_boot", ROOT/"scripts/native_df_qiskit_validation_support.py")
boot = importlib.util.module_from_spec(spec); spec.loader.exec_module(boot)
boot.package("trotterlib", ROOT/"src/trotterlib")
boot.package("trotterlib.df_trotter", ROOT/"src/trotterlib/df_trotter")
sys.path.insert(0, str(ROOT/"src"))
from trottertracks import track_d_head_tail_h4 as m
from trottertracks.c1_stage2_native import NativeBlock, NativeModel, _rotation
from trotterlib.df_trotter.ops import DiagonalEvolutionPrimitives


def context():
    blocks = []
    for i, theta in enumerate((.13, -.27, .41)):
        rotation = np.eye(8)
        rotation[:2, :2] = [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        blocks.append(NativeBlock(str(i), _rotation(rotation), DiagonalEvolutionPrimitives(0., (), ())))
    model = NativeModel("ARTIFICIAL_EIGHT_MODE", tuple(blocks), 0., "synthetic-source", "synthetic-input")
    return S(arm=S(model=model), builder=m.c3.base.StrongBuilder(relative_fusion=True))


class Tests(unittest.TestCase):
    def setUp(self):
        self.context = context()

    def join(self, mode="physical", *, between=False, same=False):
        index = 2 if same else 1
        gh, gt = m.certified_basis(self.context, 2), m.certified_basis(self.context, index)
        circuit = QuantumCircuit(9 if mode == "ordinary" else 8)
        circuit.global_phase = .37
        circuit.compose(gh, range(8), inplace=True)
        if between: circuit.x(0)
        offset = len(circuit.data)
        circuit.compose(gt.inverse(), range(8), inplace=True)
        if mode == "ordinary": circuit.crz(.19, 8, 7)
        return circuit, offset, index

    def test_full_phase_physical_and_ordinary(self):
        for mode in ("physical", "ordinary"):
            original, offset, index = self.join(mode)
            result, row = m.splice_join(original, offset, self.context, index, mode=mode)
            np.testing.assert_allclose(Operator(result).data, Operator(original).data, atol=1e-10, rtol=0)
            self.assertEqual(row["wires"], list(range(8)))
            json.dumps(row, allow_nan=False)

    def test_actual_same_basis_inverse_pair(self):
        original, offset, index = self.join(same=True)
        result, row = m.splice_join(original, offset, self.context, index, mode="physical")
        self.assertEqual(row["inserted_instructions"], 0)
        np.testing.assert_allclose(Operator(result).data, Operator(original).data, atol=1e-10, rtol=0)

    def test_analytic_gauge_once(self):
        model, builder = self.context.arm.model, self.context.builder
        builder._basis(model, model.blocks[2])
        key = (model.source_identity, model.input_identity, model.blocks[2].rotation)
        shifted = builder.cache[key].copy(); shifted.global_phase += math.pi/7
        builder.cache[key] = shifted
        original, offset, index = self.join()
        result, row = m.splice_join(original, offset, self.context, index, mode="physical")
        np.testing.assert_allclose(Operator(result).data, Operator(original).data, atol=1e-10, rtol=0)
        self.assertNotEqual(row["head_declared_phase"], 0.)

    def test_reject_nonadjacent(self):
        original, offset, index = self.join(between=True)
        with self.assertRaisesRegex(ValueError, "NONADJACENT"):
            m.splice_join(original, offset, self.context, index, mode="physical")

    def test_reject_register_mode_wires_rotation_and_gauge(self):
        original, offset, index = self.join()
        for mode in ("directional", "ordinary"):
            with self.assertRaises(ValueError): m.splice_join(original, offset, self.context, index, mode=mode)
        with self.assertRaises(ValueError): m.certified_basis(self.context, 2, wires=tuple(reversed(range(8))))
        with self.assertRaises(ValueError): m.certified_basis(self.context, 2, rotation=self.context.arm.model.blocks[1].rotation)
        shifted = m.certified_basis(self.context, 2); shifted.global_phase += .1
        with self.assertRaises(ValueError): m.certified_basis(self.context, 2, shifted)
        with self.assertRaises(ValueError): m.old.certified_basis(self.context, 2)

    def test_reject_span_with_wrong_actual_wire(self):
        original, offset, index = self.join()
        changed = QuantumCircuit(8); changed.global_phase = original.global_phase
        changed.compose(original, list(reversed(range(8))), inplace=True)
        with self.assertRaisesRegex(ValueError, "NONADJACENT"):
            m.splice_join(changed, offset, self.context, index, mode="physical")


if __name__ == "__main__": unittest.main()
