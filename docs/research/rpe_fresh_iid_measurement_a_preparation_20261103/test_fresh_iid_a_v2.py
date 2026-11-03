"""Regression checks with synthetic modules/state; no H4 preparation or RNG."""
from contextlib import ExitStack, contextmanager
import io
import json
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
import run_fresh_iid_a_v2 as runner


class PreparationReached(Exception):
    pass


@contextmanager
def synthetic_context(preparation):
    def forbidden(*args, **kwargs):
        raise AssertionError("Downstream scientific operation must not be reached")

    modules = {}
    for name in ("trotterlib", "trotterlib.df_trotter"):
        modules[name] = ModuleType(name)
        modules[name].__path__ = []

    def add(name, **attributes):
        module = ModuleType("trotterlib." + name)
        module.__dict__.update(attributes)
        modules[module.__name__] = module

    sector = SimpleNamespace(basis_indices=np.array([0]))
    add("rte_connected_cluster_cost_validation",
        load_connected_cluster_hamiltonian_snapshot=lambda path: SimpleNamespace(n_qubits=8, n_blocks=12))
    add("df_hamiltonian", PhysicalSector=SimpleNamespace(number_sector=lambda **kw: sector))
    add("df_partial_randomized_pf", df_hamiltonian_hash=lambda h: runner.H_SHA,
        split_df_hamiltonian_by_ld=lambda h, ld: "synthetic_partition")
    prepare = Mock(side_effect=preparation)
    finite = Mock(side_effect=forbidden)
    add("df_partial_s2", prepare_df_partial_s2=prepare)
    add("research_direction_full_scope", _finite_inputs=finite)
    add("finite_rte_signal_validation", _qiskit_to_openfermion_sector_permutation=forbidden,
        _array_fingerprint=lambda state: runner.STATE_FP)
    add("df_trotter.ops", one_body_diagonal_primitives=forbidden, df_squared_diagonal_primitives=forbidden)
    fixture = {"state": np.array([1.+0.j]), "basis_indices": sector.basis_indices,
               "hamiltonian_hash": np.array(runner.H_SHA)}
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, modules))
        stack.enter_context(patch.object(runner, "source_and_input_identity", return_value=[]))
        stack.enter_context(patch.object(runner.importlib.metadata, "version",
                                        side_effect=lambda name: "1.3.0" if name == "qiskit" else "synthetic"))
        load = stack.enter_context(patch.object(runner.np, "load"))
        load.return_value.__enter__.return_value = fixture
        yield prepare, finite


class CorrectedPreparationTests(unittest.TestCase):
    def test_saved_threshold_is_passed_to_actual_context_call(self):
        with synthetic_context(PreparationReached) as (prepare, finite):
            with self.assertRaises(PreparationReached):
                runner.prepare_context()
            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.args[1], "synthetic_partition")
            self.assertEqual(prepare.call_args.kwargs,
                             {"identity_policy": "extract_identity_phase", "coefficient_atol": 1e-12})
            finite.assert_not_called()

    def test_inventory_failure_preserves_actual_diagnostics_and_stops(self):
        fake = SimpleNamespace(preparation_hash="synthetic_mismatch",
                               rte_preparation=SimpleNamespace(symbolic_tail=SimpleNamespace(components=(1, 2, 3))),
                               deterministic_blocks=())
        with synthetic_context(lambda *a, **kw: fake) as (prepare, finite):
            with self.assertRaises(ValueError) as caught:
                runner.prepare_context()
            text = str(caught.exception)
            self.assertIn("preparation_hash=synthetic_mismatch", text)
            self.assertIn("components=3", text)
            self.assertIn("deterministic_blocks=0", text)
            self.assertIn("coefficient_atol=1e-12", text)
            finite.assert_not_called()

    def test_describe_is_metadata_only_and_exposes_threshold(self):
        with patch.object(runner, "prepare_context", side_effect=AssertionError("science")), \
                patch.object(runner.np.random, "default_rng", side_effect=AssertionError("RNG")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            runner.main(["--describe"])
        record = json.loads(output.getvalue())
        self.assertEqual(record["coefficient_atol"], 1e-12)
        self.assertEqual(record["trajectory_calls"], 1928)
        self.assertEqual(record["status"], "PREPARED_CORRECTED_NOT_SCIENTIFICALLY_EXECUTED")


if __name__ == "__main__":
    unittest.main()
