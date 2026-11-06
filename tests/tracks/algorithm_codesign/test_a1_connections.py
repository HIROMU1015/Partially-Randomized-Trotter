"""Invented probabilities/events and mock prices only; no science files or RNG."""
from fractions import Fraction as F
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest import TestCase, main
from unittest.mock import Mock, patch

from trottertracks.algorithm_codesign.a1_connections import (
    connect_sampler_inputs, proposed_control_banks, selected_full_prices,
)
from trottertracks.algorithm_codesign.a1_control_variates import EventMetadata


def canonical_validator():
    # Execute only this pure existing function on invented components. Do not
    # import trotterlib.__init__, quantum modules, or a scientific runner.
    path = Path(__file__).resolve().parents[3] / "src/trotterlib/rte.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_validate_components")
    tolerance = next(ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id == "PROBABILITY_ATOL")
    namespace = dict(math=math, PROBABILITY_ATOL=tolerance)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_validate_components"]


def mock_connection(probabilities=(.25, .375, .375)):
    components = tuple(NS(component_id=name, probability=p)
                       for name, p in zip(("a", "b", "c"), probabilities))
    return connect_sampler_inputs(components, NS(orders=(0, 2),
                                  order_probabilities=(.75, .25)), source="invented",
                                  validate_components=canonical_validator())


def price_fixture():
    request = NS(repetition_count=1, step_time=.02, controlled=True, ancilla_qubit=8,
                 preparation=NS(ld=3, num_system_qubits=8, coefficient_atol=0.),
                 rte_config=NS(rte_steps=1, finite_taylor_order=2))
    compiler = NS(basis_gates=("rz", "sx", "x", "cx"), backend_name=None,
                  coupling_map=None, optimization_level=2, layout_method=None,
                  routing_method=None, transpiler_seed=17, qiskit_version="1.3.0")
    selected, evolution = object(), object()
    wrappers = tuple(NS(circuit=object(), compiler_independent_fingerprint=axis)
                     for axis in ("mock-X", "mock-Y"))
    dep = NS(apply_policy=Mock(return_value=selected),
             repeated_builder=NS(build=Mock(return_value=evolution)),
             wrapper_request=Mock(side_effect=lambda **kw: NS(**kw)),
             wrapper_builder=NS(build=Mock(side_effect=wrappers)),
             compile=Mock(side_effect=(NS(rz_count=11), NS(rz_count=13))))
    return request, compiler, dep, selected, evolution, wrappers


class A1Connections(TestCase):
    def test_cdf_grid_masses_without_rng_or_input_mutation(self):
        # Fail on accidental RNG construction, even with invented inputs.
        with patch("numpy.random.PCG64", side_effect=AssertionError("no RNG")):
            c = mock_connection()
        self.assertEqual(c.label_choice_input, (.25, .375, .375))
        self.assertEqual(c.label_cdf, (.25, .625, 1.))
        self.assertEqual(c.law.labels.masses, (F(1, 4), F(3, 8), F(3, 8)))
        self.assertEqual(c.law.orders.masses, (F(3, 4), F(1, 4)))
        self.assertFalse(any(m == "trotterlib" or m.startswith("trotterlib.")
                             or m == "qiskit" or m.startswith("qiskit.")
                             for m in sys.modules))

    def test_component_fsum_normalization_preserves_generator_arithmetic(self):
        import math
        ps = (.2, .3, .5000000000000001)
        c = mock_connection(ps)
        self.assertEqual(c.component_probability_sum, math.fsum(ps))
        self.assertEqual(c.label_choice_input, tuple(p / math.fsum(ps) for p in ps))
        self.assertEqual(c.law.labels.labels, ("a", "b", "c"))

    def test_fixed_control_moments_and_exhaustive_five_strata(self):
        c = mock_connection()
        cv, order, strata = proposed_control_banks(c, law_id="mock", a="a", b="b")
        self.assertEqual(cv.means, (F(3, 16), F(1, 16), F(3, 128)))
        self.assertEqual(cv.joint_mean((1, 2)), F(3, 128))
        self.assertEqual(cv.joint_mean((0, 2)), 0)
        events = (EventMetadata(0, ("a",), 1), EventMetadata(0, ("c",), 1),
                  EventMetadata(2, ("c", "a", "a"), -1),
                  EventMetadata(2, ("a", "c", "a"), -1),
                  EventMetadata(2, ("a", "b", "a"), -1))
        patterns = tuple(strata.values((e,)) for e in events)
        self.assertEqual(len(set(patterns)), 5)
        self.assertEqual(sum(strata.pattern_mass(p) for p in patterns), 1)
        self.assertEqual(len(cv.controls), 3)
        self.assertEqual(len(order.controls), 1)

    def test_positive_input_can_lose_grid_support_no_label_substitution(self):
        c = mock_connection((.5, 2.**-60, .5))
        self.assertEqual(c.law.labels.mass("b"), 0)
        with self.assertRaises(ValueError):
            proposed_control_banks(c, law_id="mock", a="a", b="b")
        with self.assertRaises(ValueError):
            proposed_control_banks(c, law_id="mock", a="absent", b="c")

    def test_invalid_inputs_and_choice_profile_stop(self):
        for ps in ((.4, .4, .4), (-.1, .6, .5), (float("nan"), .5, .5)):
            with self.subTest(ps=ps), self.assertRaises(ValueError):
                mock_connection(ps)
        with patch("numpy.__version__", "different"), self.assertRaises(ValueError):
            mock_connection()

    def test_selected_only_full_two_axes_no_quantum_dependencies(self):
        req, compiler, dep, selected, evolution, wrappers = price_fixture()
        with patch("trottertracks.algorithm_codesign.a1_connections._price_dependencies",
                   side_effect=AssertionError("real quantum dependency used")):
            result = selected_full_prices(req, {}, compiler,
                       training_fingerprint="mock-training", _dependencies=dep)
        dep.repeated_builder.build.assert_called_once_with(
            selected, construction_policy="boundary_optimized")
        self.assertEqual(dep.compile.call_count, 2)
        self.assertEqual([r[0] for r in result], ["cosine", "sine"])
        self.assertEqual([r[2].rz_count for r in result], [11, 13])
        for call in dep.wrapper_request.call_args_list:
            self.assertIs(call.kwargs["evolution"], evolution)
            self.assertTrue(call.kwargs["include_measurement"])
        self.assertEqual(dep.apply_policy.call_args.kwargs["maximum_support_run_length"], 1)

    def test_cache_hits_still_return_two_chargeable_price_records(self):
        req, compiler, dep, *_ = price_fixture()
        cache = NS(get_or_transpile=Mock(side_effect=[(NS(rz_count=11), "x", True),
                                                    (NS(rz_count=13), "y", True)]))
        result = selected_full_prices(req, {}, compiler, training_fingerprint="mock",
                                      cache=cache, _dependencies=dep)
        dep.compile.assert_not_called()
        self.assertEqual(len(result), 2)
        self.assertTrue(all(r[4] for r in result))
        self.assertEqual([r[2].rz_count for r in result], [11, 13])

    def test_independent_one_axis_trial_prices_only_its_readout(self):
        req, compiler, dep, *_ = price_fixture()
        result = selected_full_prices(req, {}, compiler, training_fingerprint="mock",
                                      axes=("sine",), _dependencies=dep)
        self.assertEqual([r[0] for r in result], ["sine"])
        self.assertEqual(dep.compile.call_count, 1)
        self.assertEqual(dep.wrapper_builder.build.call_count, 1)
        self.assertEqual(dep.wrapper_request.call_args.kwargs["axis"], "sine")
        for axes in ((), ("cosine", "cosine"), ("other",)):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                selected_full_prices(req, {}, compiler, training_fingerprint="mock",
                                      axes=axes, _dependencies=dep)

    def test_wrong_scope_rejected_before_build(self):
        for field, value in (("optimization_level", 1), ("qiskit_version", "different")):
            req, compiler, dep, *_ = price_fixture()
            setattr(compiler, field, value)
            with self.assertRaises(ValueError):
                selected_full_prices(req, {}, compiler, training_fingerprint="mock", _dependencies=dep)
            dep.repeated_builder.build.assert_not_called()
        req, compiler, dep, *_ = price_fixture()
        req.repetition_count = 2
        with self.assertRaises(ValueError):
            selected_full_prices(req, {}, compiler, training_fingerprint="mock", _dependencies=dep)
        dep.repeated_builder.build.assert_not_called()


if __name__ == "__main__":
    main()
