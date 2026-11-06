"""Synthetic regressions only: path phase and saved-cohort admission."""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[1]


class PairedTests(unittest.TestCase):
    def test_path_dependent_chi_and_observables_for_both_sign_patterns(self):
        for angles in ((.03, -.01, -.02, .05), (-.02, .04, .01, .02)):
            scalar = -.23
            events = [SimpleNamespace(phase=1+0j, unsigned_rotation_angle=x) for x in angles]
            phase = extension.path_phase(events, scalar)
            a, b = QuantumCircuit(2), QuantumCircuit(2)
            b.rz(2*angles[0], 0); b.rzz(2*angles[1], 0, 1)
            a.rz(2*angles[2], 1); a.rzz(2*angles[3], 0, 1); a.global_phase = scalar
            target = Operator(a).data @ Operator(b).data
            self.assertAlmostEqual(target[0, 0], np.exp(1j*phase))
            self.assertNotAlmostEqual(phase, scalar)
            core = base.reference_gadget(a, b, phase)
            for axis in base.AXES:
                wrapper = base.artificial.hadamard_wrapper(core, axis).remove_final_measurements(inplace=False)
                columns = Operator(wrapper).data[:, :4].T
                actual = base.observable_from_columns(columns, 16)
                np.testing.assert_allclose(actual, base.artificial.expected_observable(target, axis), atol=1e-12, rtol=0)
            wrong = base.reference_gadget(a, b, scalar)
            wrapper = base.artificial.hadamard_wrapper(wrong, "Y").remove_final_measurements(inplace=False)
            self.assertGreater(np.max(abs(base.observable_from_columns(Operator(wrapper).data[:, :4].T, 16)
                               - base.artificial.expected_observable(target, "Y"))), 1e-3)

    def test_wrong_endpoint_cohort_or_sample_is_rejected_before_deserialization(self):
        policy = dict(master=2026110304, cohort=extension.COHORT, axis="NEAR_PRACTICAL_PR_PRICE_ONLY",
                      samples=32, events_per_sample=32, PRNG="numpy.random.PCG64", paired=False, adaptive=False)
        record = dict(sample=0, sampling=policy, JU=extension.ENDPOINT, SOURCE=extension.COHORT_SOURCE, events=[])
        for field, value in (("JU", "other-endpoint"), ("SOURCE", "other-source"), ("sample", 1),
                             ("sampling", {**policy, "paired": True})):
            with self.assertRaisesRegex(ValueError, "SAVED_COHORT_BINDING"):
                extension.saved_record({**record, field: value}, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--origin", required=True)
    args, remaining = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location("paired_runner", ROOT / "scripts/run_track_d_h4_paired_reference.py")
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    base = runner.old_runner().bootstrap(args.origin)
    from trottertracks import track_d_h4_paired_reference as extension
    unittest.main(argv=[sys.argv[0], *remaining])
