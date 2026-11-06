"""Synthetic-only regression: no molecular input, sampling, or compilation."""
import argparse
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[1]


class AdapterTests(unittest.TestCase):
    def same(self, a, b):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)

    def test_general_gadget_matches_existing_two_mode_adapter(self):
        a, b = QuantumCircuit(2), QuantumCircuit(2)
        a.rz(.3, 0); a.rzz(-.1, 0, 1); a.global_phase = -.27
        b.rz(-.6, 1); b.global_phase = .11
        old = m.artificial.reference_gadget(a, b, .49)
        self.same(Operator(m.reference_gadget(a, b, .49)).data, Operator(old).data)

    def test_control_one_preserves_scalar_and_inverse(self):
        qc = QuantumCircuit(3); qc.global_phase = .19
        qc.rz(.21, 0); qc.crz(-.38, 2, 1); qc.cz(2, 0); qc.p(-.13, 2)
        qc.rz(.08, 2)
        full = Operator(qc).data
        selected = m.control_one(qc)
        self.same(Operator(selected).data, full[4:, 4:])
        self.same(Operator(selected.inverse()).data, full[4:, 4:].conj().T)

    def test_control_one_rejects_ancilla_target(self):
        qc = QuantumCircuit(3); qc.cx(0, 2)
        with self.assertRaisesRegex(ValueError, "UNSUPPORTED_CONTROL_ONE"):
            m.control_one(qc)

    def test_dense_reference_operator_refused(self):
        with self.assertRaisesRegex(ValueError, "FORBIDDEN"):
            m.small_operator(QuantumCircuit(17))

    def test_directional_keeps_relative_fusion_and_full_controlled_path(self):
        from trottertracks.c1_stage2_native import NativeBlock, NativeModel
        from trotterlib.df_trotter.ops import one_body_diagonal_primitives
        from trottertracks.c1_h4_strong_baseline import StrongBuilder
        u = np.array([[.8, -.6j], [-.6j, .8]])
        model = NativeModel("invented", (
            NativeBlock("one", tuple(map(tuple, np.eye(2))), one_body_diagonal_primitives(np.array([.7, -.2]), 1.)),
            NativeBlock("other", tuple(map(tuple, u)), one_body_diagonal_primitives(np.array([-.3, .9]), 1.))),
            .17, "invented-source", "invented-input")
        for time in (.125, -.125):
            builder = StrongBuilder(relative_fusion=True)
            forward = ((0, time), (1, time))
            reverse = tuple(reversed(forward))
            left = builder.sequence(model, forward, optimized=True, scalar_time=0)
            right = builder.sequence(model, reverse, optimized=True, scalar_time=0)
            tail = QuantumCircuit(3); tail.crz(.27, 2, 0); tail.p(.13, 2)
            target = left.compose(tail).compose(right)
            adapted = QuantumCircuit(3)
            adapted.compose(m.control_one(left), range(2), inplace=True)
            adapted.compose(tail, inplace=True)
            adapted.compose(m.directional_head(builder, model, reverse), inplace=True)
            self.same(Operator(adapted).data, Operator(target).data)
            self.assertTrue(builder.fusion_records)
            self.assertTrue(all("phase_correction" in row for row in builder.fusion_records))

    def test_observable_columns_keep_complex_cross_terms(self):
        qc = QuantumCircuit(2); qc.h(1); qc.cp(.43, 1, 0); qc.sdg(1); qc.h(1)
        op = Operator(qc).data
        actual = m.observable_from_columns(op[:, :2].T, 2)
        z = np.diag([1, 1, -1, -1])
        self.same(actual, (op.conj().T @ z @ op)[:2, :2])

    def test_reference_isometry_layout_and_phase(self):
        a, b = QuantumCircuit(2), QuantumCircuit(2)
        a.rz(.31, 0); a.global_phase = -.2
        b.rz(-.27, 1); b.global_phase = .17
        chi = -.05
        core = m.reference_gadget(a, b, chi)
        full = QuantumCircuit(5); full.h(4); full.compose(core, inplace=True)
        cols = Operator(full).data[:, :4].T
        A, B = Operator(a).data, Operator(b).data
        records = []
        def same(x, y, label): self.same(x, y); records.append(label)
        m.reference_column_check(cols, B[0,0]*A.conj().T/np.sqrt(2),
            np.exp(1j*chi)*A.conj().T[0,0]*B/np.sqrt(2), same, "invented")
        self.assertEqual(len(records), 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--origin", required=True)
    args, remaining = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location("track_d_h4_runner", ROOT/"scripts/run_track_d_h4_reference.py")
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    m = runner.bootstrap(Path(args.origin))
    unittest.main(argv=[sys.argv[0], *remaining])
