"""Artificial coefficient/data tests and mocked legacy bridges only.

No QuantumCircuit, transpile, Operator, Statevector, molecule, fit, reference
matrix, scientific snapshot decode or sampling call is allowed in this suite.
The selected existing modules are loaded behind passive package/Qiskit doubles
to avoid the library's broad eager __init__ imports. Their formulas, primitive
type, registry, component types and diagonal emitter remain the real SOURCE.
"""
from contextlib import contextmanager
from dataclasses import replace
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy  # one lifetime; only legacy coefficient arithmetic uses it

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from trottertracks import c1_general_k_adapter as adapter


def frame(K=((2., -4.), (-4., 6.)), **updates):
    n = len(K)
    args = dict(frame_id="F0", group_id="G0", original_index=0, K=K,
                orbital_rotation=tuple(tuple(float(a == b) for b in range(n))
                                       for a in range(n)),
                source_identity="artificial-source", input_identity="artificial-input")
    args.update(updates)
    return adapter.GeneralKFrame(**args)


def model_args():
    return dict(one_body_tensor=((.2, .125), (.125, -.3)),
                diag_coulomb_mats=(((2., -4.), (-4., 6.)),),
                orbital_rotations=(((0., 1j), (1j, 0.)),), constant=7.,
                z_representation=False, one_body_role="ffsim_corrected_kappa",
                frame_ids=("F0",), group_ids=("G0",),
                source_identity="mock-ffsim:" + adapter.FFSIM_SOURCE,
                input_identity="artificial-input", enable_general_k=True)


class Gate:
    def __init__(self, name, angle):
        self.name, self.params = name, [angle]

    def control(self, count):
        if count != 1:
            raise AssertionError("Only the existing single control bridge is tested")
        return Gate("c-" + self.name, self.params[0])


class Recorder:
    def __init__(self):
        self.calls = []
        self.global_phase = 0.0

    def append(self, gate, qubits):
        self.calls.append((gate.name, tuple(gate.params), tuple(qubits)))

    def rz(self, angle, qubit):
        self.calls.append(("rz", (angle,), (qubit,)))

    def rzz(self, angle, left, right):
        self.calls.append(("rzz", (angle,), (left, right)))


def package(name, path=None, **attrs):
    result = ModuleType(name)
    if path is not None:
        result.__path__ = [str(path)]
    result.__dict__.update(attrs)
    return result


@contextmanager
def existing_data_bridges():
    forbidden = Mock(side_effect=AssertionError("Scientific/build entry is forbidden"))
    stubs = {
        "trotterlib": package("trotterlib", ROOT / "src/trotterlib"),
        "trotterlib.df_trotter": package("trotterlib.df_trotter",
                                        ROOT / "src/trotterlib/df_trotter"),
        "trotterlib.df_trotter.decompose": package(
            "trotterlib.df_trotter.decompose", diag_hermitian=forbidden),
        "qiskit": package("qiskit", QuantumCircuit=forbidden,
                           QuantumRegister=forbidden, transpile=forbidden),
        "qiskit.circuit": package("qiskit.circuit", ParameterExpression=type("Parameter", (), {})),
        "qiskit.circuit.library": package(
            "qiskit.circuit.library", PhaseGate=lambda a: Gate("p", a),
            RZGate=lambda a: Gate("rz", a), RZZGate=lambda a: Gate("rzz", a)),
        "qiskit.quantum_info": package("qiskit.quantum_info", Operator=forbidden,
                                       Statevector=forbidden),
    }
    with patch.dict(sys.modules, stubs):
        ops = importlib.import_module("trotterlib.df_trotter.ops")
        tail = importlib.import_module("trotterlib.df_rte_tail")
        specs = importlib.import_module("trotterlib.df_rte_circuit")
        BasisChangeOperation = importlib.import_module("trotterlib.rte").BasisChangeOperation

        def describe_mock(operations):
            # Real registry/hash and dataclass, with gate fingerprinting mocked.
            # Production describe_basis_change_operations can evaluate Operator;
            # that evaluation is outside this technical test's authorization.
            return tuple(BasisChangeOperation(
                name=gate.name, qubits=qubits,
                parameters=tuple(str(x) for x in gate.params),
                matrix_sha256="mock_fingerprint_no_matrix_evaluation")
                for gate, qubits in operations)

        with (patch.object(ops, "U_to_qiskit_ops_jw_givens", forbidden),
              patch.object(ops.np.linalg, "eigh", forbidden),
              patch.object(ops.np.linalg, "norm", forbidden),
              patch.object(tail, "describe_basis_change_operations", describe_mock)):
            yield SimpleNamespace(ops=ops, tail=tail, specs=specs)
        forbidden.assert_not_called()


class GeneralKTests(unittest.TestCase):
    def tearDown(self):
        for name in ("ffsim", "qiskit", "pyscf", "openfermion", "trotterlib"):
            self.assertNotIn(name, sys.modules)

    def test_import_is_passive_and_general_route_is_explicit(self):
        args = model_args()
        for enabled in (False, None, 1, "True"):
            with self.subTest(enabled=enabled), self.assertRaises(ValueError):
                adapter.from_ffsim_number_arrays(**{**args, "enable_general_k": enabled})
        model = adapter.from_ffsim_number_arrays(**args)
        self.assertIsInstance(model.frames[0], adapter.GeneralKFrame)
        self.assertFalse(hasattr(model.frames[0], "eta"))
        self.assertFalse(hasattr(model.frames[0], "lam"))

    def test_real_symmetric_validation_and_immutable_input_copy(self):
        raw = [[2., -4.], [-4., 6.]]
        result = frame(raw)
        raw[0][0] = 100
        self.assertEqual(result.K, ((2., -4.), (-4., 6.)))
        invalid = ([], [[1, 2]], [[1, 2], [3, 1]], [[1, float("nan")], [float("nan"), 1]],
                   [[float("inf")]], [[1 + 0j]], [[True]], [[[1]]])
        for K in invalid:
            with self.subTest(K=K), self.assertRaises(ValueError):
                adapter.general_k_coefficients(K)
        with self.assertRaises(ValueError):
            frame(orbital_rotation=((1.,),))
        with self.assertRaises(ValueError):
            frame(orbital_rotation=((1., 0.), (0., complex(0, float("nan")))))
        for kwargs in ({"frame_id": ""}, {"source_identity": ""},
                       {"original_index": True}, {"original_index": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                frame(**kwargs)

    def test_general_signed_density_coefficients(self):
        self.assertEqual(adapter.general_k_coefficients(((2., -4.), (-4., 6.))),
                         (((), 2.), ((0,), 1.), ((1,), -1.), ((0, 1), -2.)))
        self.assertEqual(adapter.general_k_coefficients(
            ((2., 1., -3.), (1., -4., 2.), (-3., 2., 6.))),
            (((), 2.), ((0,), 0.), ((1,), .5), ((2,), -2.5),
             ((0, 1), .5), ((0, 2), -1.5), ((1, 2), 1.)))

    def test_diagonal_zero_sparse_and_nonfinite_arithmetic(self):
        self.assertEqual(adapter.general_k_coefficients(((2., 0.), (0., -2.))),
                         (((), 0.), ((0,), -1.), ((1,), 1.), ((0, 1), 0.)))
        with existing_data_bridges():
            zero = adapter.diagonal_primitives(frame(((0., 0.), (0., 0.))), .25)
            self.assertEqual((zero.global_phase, zero.rz, zero.rzz), (0., (), ()))
            tiny = adapter.diagonal_primitives(frame(((1e-15,),)), 1.)
            self.assertEqual(tiny.rz, ((0, -1e-15),))
            self.assertEqual(adapter.diagonal_primitives(frame(), 0.).rz, ())
            with self.assertRaises(ValueError):
                adapter.diagonal_primitives(frame(), float("nan"))
            with self.assertRaises(ValueError):
                adapter.diagonal_primitives(frame(), 1e308)
        with self.assertRaises(ValueError):
            adapter.general_k_coefficients(((1e308, 1e308), (1e308, 1e308)))

    def test_rank1_degenerates_to_existing_square_formulas(self):
        eta, lam = (.5, -2., 1.25), -.75
        K = tuple(tuple(lam * a * b for b in eta) for a in eta)
        with existing_data_bridges() as legacy:
            expected = legacy.tail.exact_df_diagonal_coefficients(eta, lam)
            self.assertEqual(adapter.general_k_coefficients(K), expected)
            for tau in (-.125, 0., .5):
                actual = adapter.diagonal_primitives(frame(K), tau)
                expected = legacy.ops.df_squared_diagonal_primitives(eta, lam, tau)
                # The legacy primitive list retains zero RZZ placeholders;
                # its emitter also removes exact zeros. Compare emitted data.
                self.assertEqual(actual, replace(expected, rzz=tuple(
                    item for item in expected.rzz if item[2] != 0.0)))
            # Existing square data type and constructor retain their old meaning.
            old = importlib.import_module("trotterlib.df_trotter.model").DFBlock
            square = old(U_ops=(), eta=eta, lam=lam)
            self.assertEqual((square.eta, square.lam), (eta, lam))
            self.assertFalse(hasattr(square, "K"))

    def test_ffsim_half_factor_spin_mapping_and_raw_U(self):
        model = adapter.from_ffsim_number_arrays(**model_args())
        f = model.frames[0]
        self.assertEqual(f.K, ((1., 1., -2., -2.), (1., 1., -2., -2.),
                               (-2., -2., 3., 3.), (-2., -2., 3., 3.)))
        coefficients = dict(adapter.general_k_coefficients(f.K))
        self.assertEqual(coefficients[(0, 1)], .5)  # same spatial, opposite spin
        self.assertEqual(coefficients[(0, 3)], -1.)
        self.assertEqual(f.interleaved_to_block_permutation, (0, 2, 1, 3))
        self.assertEqual(f.orbital_rotation[0], (0j, 0j, 1j, 0j))
        self.assertEqual(f.orbital_rotation[1], (0j, 0j, 0j, 1j))
        request = f.gaussian_request()
        self.assertEqual(request["V"], f.orbital_rotation)
        self.assertEqual(request["application_order"],
                         ("basis_inverse", "central", "basis_forward"))
        self.assertEqual(request["independent_CAR_validation"], "pending")
        self.assertIn("V.T", request["helper_argument"])
        with self.assertRaises(ValueError):
            replace(f, K=tuple(tuple(2 * x for x in row) for row in f.K))

    def test_number_representation_and_kappa_role_are_not_inferred(self):
        args = model_args()
        for z in (True, None, 0, "number"):
            with self.subTest(z=z), self.assertRaises(ValueError):
                adapter.from_ffsim_number_arrays(**{**args, "z_representation": z})
        with self.assertRaises(ValueError):
            adapter.from_ffsim_number_arrays(**{**args, "one_body_role": "original_h"})
        for kwargs in ({"representation": "z"}, {"spin_order": "alpha_then_beta"},
                       {"coefficient_convention": "without_half"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                frame(**kwargs)
        with self.assertRaises(ValueError):
            adapter.from_ffsim_number_arrays(**{**args, "group_ids": ()})
        with self.assertRaises(ValueError):
            adapter.from_ffsim_number_arrays(**{**args, "diag_coulomb_mats": (((1.,),),)})
        with self.assertRaises(ValueError):
            adapter.from_ffsim_number_arrays(**{**args, "constant": float("inf")})

    def test_numpy_export_array_boundary_is_copied_without_fit(self):
        args = model_args()
        baseline = adapter.from_ffsim_number_arrays(**args)
        arrays = {name: numpy.asarray(args[name]) for name in
                  ("one_body_tensor", "diag_coulomb_mats", "orbital_rotations")}
        imported = adapter.from_ffsim_number_arrays(**{**args, **arrays})
        self.assertEqual(imported, baseline)
        arrays["diag_coulomb_mats"][0, 0, 0] = 99.
        arrays["orbital_rotations"][0, 0, 0] = 99j
        self.assertEqual(imported, baseline)
        with self.assertRaises(ValueError):
            adapter.general_k_coefficients(numpy.asarray([[True]]))
        with self.assertRaises(ValueError):
            adapter.general_k_coefficients(numpy.asarray([[1 + 0j]]))
        # No unapproved Hermiticity tolerance or projection is introduced.
        # A future scientific gate must reject a physically unsuitable kappa.
        pending = adapter.from_ffsim_number_arrays(**{
            **args, "one_body_tensor": ((1., 1j), (1j, 1.))})
        self.assertEqual(pending.one_body_tensor[0][2], 1j)
        self.assertEqual(pending.reference_input()["one_body_hermiticity"],
                         "pending_separate_scientific_gate")

    def test_scalar_and_kappa_stay_outside_each_frame(self):
        model = adapter.from_ffsim_number_arrays(**model_args())
        other = adapter.from_ffsim_number_arrays(**{
            **model_args(), "one_body_tensor": ((9., 0.), (0., 11.)), "constant": -5.})
        self.assertEqual(model.frames, other.frames)
        self.assertEqual(model.one_body_tensor[0][2], .125)
        self.assertEqual(model.one_body_tensor[0][1], 0.)
        self.assertEqual(model.constant, 7.)
        self.assertNotEqual(model.cache_identity, other.cache_identity)
        with existing_data_bridges():
            self.assertEqual(adapter.diagonal_primitives(model.frames[0], .5),
                             adapter.diagonal_primitives(other.frames[0], .5))
        with self.assertRaises(ValueError):
            replace(model, one_body_tensor=((0., 1.), (0., 0.)))

    def test_existing_emitter_keeps_controlled_relative_phase(self):
        with existing_data_bridges() as legacy:
            controlled, plain = Recorder(), Recorder()
            with patch.object(legacy.ops, "append_diagonal_primitives",
                              wraps=legacy.ops.append_diagonal_primitives) as emitter:
                p = adapter.append_general_k_diagonal(
                    controlled, frame(), .25, controlled=True, ancilla_qubit=2,
                    enable_general_k=True)
                emitter.assert_called_once()
            self.assertEqual((p.global_phase, p.rz, p.rzz),
                             (-.5, ((0, .5), (1, -.5)), ((0, 1, -1.),)))
            self.assertEqual(controlled.calls,
                             [("c-rzz", (-1.,), (2, 0, 1)),
                              ("c-rz", (.5,), (2, 0)),
                              ("c-rz", (-.5,), (2, 1)), ("p", (-.5,), (2,))])
            self.assertEqual(controlled.global_phase, 0.)
            adapter.append_general_k_diagonal(plain, frame(), .25, controlled=False,
                                              enable_general_k=True)
            self.assertEqual(plain.global_phase, -.5)
            self.assertFalse(any(call[0] == "p" for call in plain.calls))
            for kwargs in ({"controlled": True}, {"controlled": True, "ancilla_qubit": 1},
                           {"controlled": False, "ancilla_qubit": 2}, {"controlled": 1}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    adapter.append_general_k_diagonal(Recorder(), frame(), .25,
                                                      enable_general_k=True, **kwargs)
            with self.assertRaises(ValueError):
                adapter.append_general_k_diagonal(Recorder(), frame(), .25, controlled=False)

    def test_original_order_group_rank_and_namespaced_identity(self):
        args = model_args()
        args.update(diag_coulomb_mats=args["diag_coulomb_mats"] * 2,
                    orbital_rotations=(args["orbital_rotations"][0], ((1., 0.), (0., 1.))),
                    frame_ids=("z-last", "a-first"), group_ids=("G7", "G2"))
        model = adapter.from_ffsim_number_arrays(**args)
        self.assertEqual(tuple(f.frame_id for f in model.frames), ("z-last", "a-first"))
        self.assertEqual(tuple(f.group_id for f in model.frames), ("G7", "G2"))
        self.assertEqual(model.original_outer_rank, 2)
        self.assertFalse(hasattr(model, "auxiliary_square_rank"))
        self.assertTrue(model.cache_identity.startswith(adapter.SCHEMA + ":"))
        for updated in (replace(model, input_identity="other", frames=tuple(
                replace(f, input_identity="other") for f in model.frames)),
                        replace(model, source_identity="other", frames=tuple(
                replace(f, source_identity="other") for f in model.frames))):
            self.assertNotEqual(model.cache_identity, updated.cache_identity)
        self.assertNotEqual(model.frames[0].frame_hash,
                            replace(model.frames[0], group_id="other").frame_hash)
        with self.assertRaises(ValueError):
            replace(model, frames=tuple(reversed(model.frames)))
        with self.assertRaises(ValueError):
            replace(model, frames=(model.frames[0], replace(model.frames[1], frame_id="z-last")))

    def test_existing_registry_components_and_deterministic_sidecar(self):
        f = frame()
        with existing_data_bridges() as legacy:
            registry = legacy.tail.DFBasisRegistry()
            operations = ((Gate("rz", .125), (0,)),)
            bound = adapter.bind_frame(f, order_index=3, basis_registry=registry,
                                       runtime_operations=operations, enable_general_k=True)
            self.assertEqual(bound.order_index, 3)
            self.assertEqual(bound.block_kind, "general_k_fragment")
            self.assertEqual(bound.validation_status, "NOT_NUMERICALLY_VALIDATED")
            self.assertIs(bound.frame, f)
            self.assertEqual(registry.operations(f.logical_basis_id), operations)
            self.assertTrue(bound.basis_definition.basis_id.startswith("c1-general-k-basis:"))
            components = bound.diagonal_components()
            self.assertTrue(all(isinstance(c, legacy.tail.DFDiagonalPauliComponent)
                                for c in components))
            self.assertEqual(tuple(c.coefficient for c in components), (2., 1., -1., -2.))
            specs = bound.component_specs()
            self.assertIsInstance(specs[0], legacy.specs.DFRTEIdentityCircuitSpec)
            self.assertEqual(specs[0].controlled_action, "relative_ancilla_phase")
            self.assertEqual(tuple(s.diagonal_pauli_support for s in specs),
                             ((), (0,), (1,), (0, 1)))
            self.assertTrue(all(s.coefficient_abs > 0 for s in specs))
            self.assertEqual(specs[-1].coefficient_sign, -1)
            zero = frame(((0., 0.), (0., 0.)))
            zero_bound = adapter.bind_frame(zero, order_index=0, basis_registry=registry,
                                            runtime_operations=operations, enable_general_k=True)
            self.assertEqual(zero_bound.component_specs(), ())
            with self.assertRaises(ValueError):
                adapter.bind_frame(f, order_index=0, basis_registry=registry,
                                   runtime_operations=((Gate("rz", .5), (0,)),),
                                   enable_general_k=True)
            with self.assertRaises(ValueError):
                adapter.bind_frame(f, order_index=0, basis_registry=registry,
                                   runtime_operations=operations)
            with self.assertRaises(ValueError):
                replace(bound, order_index=-1)

    def test_policy_is_pending_and_reference_data_precede_Z_adapter(self):
        model = adapter.from_ffsim_number_arrays(**model_args())
        reference = model.reference_input()
        self.assertEqual(reference["frames"][0]["J"], model.frames[0].spatial_J)
        self.assertEqual(reference["frames"][0]["U"], model.frames[0].spatial_orbital_rotation)
        self.assertIsNone(reference["frames"][0]["raw_K"])
        self.assertEqual(reference["independent_reference_status"], "not_evaluated")
        for name in ("lambda_r", "L_D", "partition", "sampling_law", "finite_target"):
            self.assertFalse(hasattr(model, name))
        for extra in ({"ld": 1}, {"identity_policy": "extract_identity_phase"}):
            with self.subTest(extra=extra), self.assertRaises(TypeError):
                adapter.from_ffsim_number_arrays(**model_args(), **extra)
        with self.assertRaises(ValueError):
            frame().interleaved_to_block_permutation


if __name__ == "__main__":
    unittest.main()
