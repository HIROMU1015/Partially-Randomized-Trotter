"""Invented labels and manually written bits only. No science files or RNG.

No quantum simulation/build/compile and no saved scientific statistics are
used. Values here test algebra/interfaces, not PR variance reduction.
"""
import ast
from bisect import bisect_right
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import unittest

from trottertracks.algorithm_codesign.a1_control_variates import (
    Batch, CategoricalLaw, ControlBank, EventMetadata, FrozenCV, LocalControl,
    Measurement, OccurrenceLaw, Scope, TrialCost, FULL_SCOPE,
    account_costs, evaluation_inputs, fit_and_freeze, law_from_choice_cdf,
    raw_mean, require_disjoint, stratified_mean, bind_wrapper_measurement,
)
from trottertracks.algorithm_codesign.return_aggregation import dyadic_index


def fixture(axis="X"):
    law = OccurrenceLaw(CategoricalLaw((0, 2), (F(3, 4), F(1, 4)), "invented order CDF"),
                        CategoricalLaw(("a", "b"), (F(1, 4), F(3, 4)), "invented label CDF"))
    bank = ControlBank("mock-law", (law,), (LocalControl(0, 0, ((0, "a"),)),
                      LocalControl(0, 2, ((0, "a"), (1, "b")))))
    scope = Scope("mock-target", bank.law_id, "mock-full-wrapper", "mock-round", axis, F(2))
    return scope, bank


def event(labels, order=0):
    return EventMetadata(order, tuple(labels), complex((-1)**(order//2)))


def batch(stage="train", axis="X", stream=None):
    scope, _ = fixture(axis)
    stream = stream or stage
    rows = ((event(("a",)), 0, F(1)), (event(("b",)), 1, F(2)),
            (event(("a", "b", "a"), 2), 1, F(-1)), (event(("b", "b", "b"), 2), 0, F(1)))
    return Batch(scope, stage, tuple(Measurement(stream, str(i), (e,), bit, w, "mock-path"+str(i), "mock-wrapper"+str(i))
                                   for i, (e, bit, w) in enumerate(rows)))


class A1Adapters(unittest.TestCase):
    def test_choice_cdf_grid_boundaries_and_existing_dyadic_mapping(self):
        law = law_from_choice_cdf(("a", "b", "c"), (.25, .625, 1.), source="mock actual CDF")
        self.assertEqual(law.masses, (F(1, 4), F(3, 8), F(3, 8)))
        # Reuse the existing deterministic bit-to-index function; no RNG.
        for k in range(8):
            self.assertEqual(dyadic_index(law.masses, 3, k), bisect_right((.25, .625, 1.), k/8))
        c = .1
        law = law_from_choice_cdf(("a", "b"), (c, 1.), source="mock rounded CDF")
        self.assertNotEqual(law.masses[0], F(1, 10))
        self.assertLess(law.masses[0] - F.from_float(c), F(1, 2**53))
        self.assertGreaterEqual(law.masses[0], F.from_float(c))

    def test_bad_cdf_and_law(self):
        for cdf in ((.8, .7, 1.), (-.1, 1.), (.5, .9), (float("nan"), 1.), (float("inf"), 1.)):
            with self.subTest(cdf=cdf), self.assertRaises(ValueError):
                law_from_choice_cdf(tuple(range(len(cdf))), cdf, source="mock")
        for labels, masses in ((("a", "a"), (F(1, 2), F(1, 2))), (("a",), (F(2),))):
            with self.assertRaises(ValueError): CategoricalLaw(labels, masses, "mock")
        with self.assertRaises(TypeError): CategoricalLaw(("a",), (float("inf"),), "mock")

    def test_local_controls_center_and_joint_moments_by_small_classical_sum(self):
        _, bank = fixture()
        means = [F(0), F(0)]
        law = bank.occurrences[0]
        for n, qn in zip(law.orders.labels, law.orders.masses):
            for labels in product(law.labels.labels, repeat=n+1):
                mass = qn
                for label in labels: mass *= law.labels.mass(label)
                for i, g in enumerate(bank.values((event(labels, n),))): means[i] += mass*g
        self.assertEqual(tuple(means), (F(3, 16), F(3, 64)))
        self.assertEqual(tuple(means), bank.means)
        self.assertEqual(bank.joint_mean((0, 1)), 0)
        self.assertEqual(bank.joint_mean((1, 1)), bank.means[1])
        self.assertEqual(sum(bank.pattern_mass(p) for p in product((0, 1), repeat=2)), 1)

    def test_multiple_occurrence_ordering_and_repeated_position_mean(self):
        _, bank = fixture()
        c = (LocalControl(0, 2, ((0, "a"),)), LocalControl(0, 2, ((0, "b"),)),
             LocalControl(1, 0, ((0, "a"),)))
        other = ControlBank("mock", bank.occurrences*2, c)
        self.assertEqual(other.joint_mean((0, 1)), 0)
        self.assertEqual(other.joint_mean((0, 2)), other.means[0]*other.means[2])
        self.assertEqual(other.values((event(("a", "b", "a"), 2), event(("a",)))), (1, 0, 1))

    def test_rte_metadata_application_order_and_phase_preserved(self):
        e = SimpleNamespace(taylor_order=2, selected_component_ids=("a", "b", "a"),
                            product_component_ids=("a", "b"), rotation_component_id="a", phase=-1+0j)
        self.assertEqual(EventMetadata.from_rte_event(e), event(("a", "b", "a"), 2))
        e.selected_component_ids = ("b", "a", "a")
        with self.assertRaises(ValueError): EventMetadata.from_rte_event(e)
        with self.assertRaises(ValueError): EventMetadata(2, ("a", "b", "a"), 1)

    def test_existing_readout_contract_static_no_quantum_import(self):
        path = Path(__file__).resolve().parents[3] / "src/trotterlib/rpe_hadamard_interrogation.py"
        tree = ast.parse(path.read_text())
        assignments = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign)
                       and isinstance(n.targets[0], ast.Name)}
        self.assertEqual(ast.literal_eval(assignments["RPE_HADAMARD_BIT_VALUE_MAPPING"]), ((0, 1), (1, -1)))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_build_validated")
        readout = ast.get_source_segment(path.read_text(), method)
        self.assertIn('request.axis == "sine"', readout)
        self.assertLess(readout.index("circuit.sdg(ancilla)"), readout.rindex("circuit.h(ancilla)"))

    def test_zero_lambda_both_axes_weights_and_no_second_phase_sign(self):
        for axis in ("X", "Y"):
            _, bank = fixture(axis)
            train, production = batch(axis=axis), batch("prod", axis)
            cv = FrozenCV(bank, train, (F(0), F(0)))
            self.assertEqual(cv.apply(production), (F(1), F(-2), F(1), F(1)))
            self.assertEqual(sum(cv.apply(production))/4, raw_mean(production, bank))

    def test_bind_actual_wrapper_metadata_mock_both_axes(self):
        for axis, name, component in (("X", "cosine", "real"), ("Y", "sine", "imaginary")):
            scope, _ = fixture(axis)
            wrapper = SimpleNamespace(wrapped_trajectory_fingerprint="path1", axis=name,
                signal_component=component, bit_value_mapping=((0, 1), (1, -1)),
                circuit_scope=FULL_SCOPE, include_measurement=True, state_preparation_included=False,
                additional_control_applied=False, wrapped_evolution_already_controlled=True,
                wrapper_fingerprint="wrapper1")
            kwargs = dict(trajectory_id="path1", events=(event(("a",)),), stream_id="s", trial_id="0", bit=1, weight=F(2))
            record = bind_wrapper_measurement(scope, wrapper, **kwargs)
            self.assertEqual((record.raw, record.wrapper_id), (-2, "wrapper1"))
            with self.assertRaises(ValueError): bind_wrapper_measurement(scope, wrapper, **dict(kwargs, trajectory_id="other"))
            wrapper.bit_value_mapping = ((0, -1), (1, 1))
            with self.assertRaises(ValueError): bind_wrapper_measurement(scope, wrapper, **kwargs)

    def test_fitted_coefficients_frozen_and_from_training_only(self):
        _, bank = fixture()
        train = batch()
        cv = fit_and_freeze(bank, train)
        self.assertTrue(all(isinstance(x, F) for x in cv.coefficients))
        with self.assertRaises(FrozenInstanceError): cv.coefficients = (F(0), F(0))
        before = cv.coefficients
        cv.apply(batch("prod"))
        self.assertEqual(cv.coefficients, before)
        with self.assertRaises(ValueError): fit_and_freeze(bank, batch("cal"))

    def test_fit_matches_manually_specified_linear_mock_response(self):
        scope, bank = fixture()
        bank = replace(bank, controls=bank.controls[:1])
        records = tuple(Measurement("linear-train", str(i), (event((label,)),), bit, F(1), "path", "wrapper")
                        for i, (label, bit) in enumerate((("a", 0), ("b", 1), ("a", 0), ("b", 1))))
        cv = fit_and_freeze(bank, Batch(scope, "train", records))
        self.assertAlmostEqual(float(cv.coefficients[0]), 2., places=12)

    def test_data_reuse_stage_and_scope_mismatch(self):
        _, bank = fixture()
        cv = FrozenCV(bank, batch(), (F(1), F(-2)))
        with self.assertRaises(ValueError): cv.apply(batch())
        with self.assertRaises(ValueError): cv.apply(batch("cal", stream="train"))
        with self.assertRaises(ValueError): evaluation_inputs(cv, batch("cal"), batch("prod", stream="cal"), mean_error_bounds=(0, 0))
        with self.assertRaises(ValueError): cv.apply(batch("prod", "Y"))
        with self.assertRaises(ValueError): cv.apply(replace(batch("prod"), scope=replace(batch().scope, target_id="another")))
        outputs = evaluation_inputs(cv, batch("cal"), batch("prod"), mean_error_bounds=(F(1, 100), F(1, 200)))
        self.assertEqual(outputs[0].centering_bias, F(1, 50))
        self.assertEqual([x.stage for x in outputs], ["cal", "prod"])

    def test_conservative_range_contains_all_mock_outputs(self):
        _, bank = fixture()
        cv = FrozenCV(bank, batch(), (F(3), F(-5)))
        for values in product((0, 1), repeat=2):
            for z in (-2, 2):
                v = z - sum(c*(g-nu) for c, g, nu in zip(cv.coefficients, values, bank.means))
                self.assertLessEqual(cv.output_range.lo, v)
                self.assertGreaterEqual(cv.output_range.hi, v)
        self.assertGreater(cv.output_range.hi-cv.output_range.lo, 4)

    def test_invalid_support_weight_and_nonfinite(self):
        _, bank = fixture()
        for events in ((event(("missing",)),), (), (event(("a",)), event(("a",)))):
            with self.assertRaises(ValueError): bank.values(events)
        with self.assertRaises(ValueError): replace(batch(), records=(replace(batch().records[0], weight=F(3)),))
        with self.assertRaises(ValueError): replace(batch(), records=(batch().records[0],)*2)
        with self.assertRaises(TypeError): FrozenCV(bank, batch(), (float("nan"), F(0)))
        with self.assertRaises(ValueError): raw_mean(replace(batch(), scope=replace(batch().scope, law_id="wrong")), bank)
        with self.assertRaises(ValueError): LocalControl(0, 0, ((1, "a"),))

    def test_strong_strata_same_target_and_complete_mass(self):
        _, bank = fixture()
        cells = []
        for i, (key, e, bit) in enumerate((((1, 0), event(("a",)), 0),
                                        ((0, 1), event(("a", "b", "a"), 2), 1),
                                        ((0, 0), event(("b",)), 1))):
            b = Batch(batch().scope, "prod", (Measurement("cell"+str(i), "0", (e,), bit, F(1), "mock-path", "mock-wrapper"),))
            cells.append((key, b))
        self.assertEqual(stratified_mean(bank, cells), F(-5, 8))
        with self.assertRaises(ValueError): stratified_mean(bank, cells[:-1])
        with self.assertRaises(ValueError): stratified_mean(bank, cells+[cells[0]])
        wrong = cells[:-1]+[(cells[-1][0], replace(cells[-1][1], scope=replace(batch().scope, round_id="other")))]
        with self.assertRaises(ValueError): stratified_mean(bank, wrong)

    def test_costs_all_stages_no_double_count_unknown_T_retained(self):
        scope, _ = fixture()
        rows = tuple(TrialCost(scope, stage, "0", stage, FULL_SCOPE,
                              (None, F(2), F(3), F(4), F(5)), F(1), F(2), 30)
                     for stage in ("train", "cal", "prod"))
        total = account_costs(rows)
        self.assertEqual(total["quantum_total"], (None, F(6), F(9), F(12), F(15)))
        self.assertEqual(total["classical_setup_seconds"], 3)
        self.assertEqual(total["classical_online_seconds"], 6)
        self.assertEqual(account_costs(rows[-1:])["quantum_total"], rows[-1].quantum)  # raw needs no train
        with self.assertRaises(ValueError): account_costs(rows+(rows[0],))
        with self.assertRaises(ValueError): account_costs((replace(rows[0], full_scope="central RTE only"),))
        with self.assertRaises(ValueError): account_costs((replace(rows[0], quantum=(0, 0, -1, 0, 0)),))


if __name__ == "__main__":
    unittest.main()
