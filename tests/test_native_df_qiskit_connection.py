"""Small actual-Qiskit builders versus the declared theoretical construction."""
from fractions import Fraction as F
import itertools
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import native_df_qiskit_validation_support as s

NATIVE = None


def setUpModule():
    global NATIVE
    NATIVE = s.bootstrap()
    # Guards are installed after real module import. They are test guardrails,
    # not an OS sandbox, and do not alter mathematical builder implementations.
    import numpy as np
    import qiskit
    import trotterlib.df_trotter.decompose as decomposition
    import trotterlib.rte as rte
    import trotterlib.df_rte_circuit as requests
    import trotterlib.rte_compiled_cost as costs
    def forbidden(*args, **kwargs):
        raise AssertionError("Molecular I/O, sampling, decomposition or transpile is outside this test")
    np.load = np.save = np.savez = np.savez_compressed = forbidden
    qiskit.transpile = costs.transpile = forbidden
    decomposition.df_decompose_from_integrals = decomposition._import_low_rank_decomposition = forbidden
    for module in (rte, requests):
        for name in ("sample_rte_events", "iter_sample_rte_events"):
            if hasattr(module, name): setattr(module, name, forbidden)


class ConnectionChecks(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from qiskit.quantum_info import Operator
        from trotterlib import native_df_gate_model as model
        self.np, self.Operator, self.m = np, Operator, model

    def same_unitary(self, a, b):
        residual = float(self.np.max(self.np.abs(self.Operator(a).data - self.Operator(b).data)))
        self.assertLess(residual, 3e-12)
        return residual

    def connection(self, qc, ir, label, check_unitary=True):
        expected = s.circuit_from_ir(ir, qc.num_qubits)
        a, b = s.word(qc), s.word(expected)
        self.assertEqual(len(a), len(b), label)
        for actual, theoretical in zip(a, b):
            self.assertEqual(actual[:2], theoretical[:2], label)
            self.np.testing.assert_allclose(actual[2], theoretical[2], atol=2e-15, rtol=2e-15)
        cost = s.qiskit_cost(qc)
        self.assertEqual(cost, self.m.count(ir), label)
        record = {"label": label, "raw_word_match": True, "cost": cost.as_dict()}
        if check_unitary:
            record["unitary_max_abs"] = self.same_unitary(qc, expected)
            record["fixed_lowering_unitary_max_abs"] = self.same_unitary(qc, s.fixed_decompose(qc))
        s.OBSERVATIONS.append(record)

    def test_actual_qiskit_gate_definitions_and_fixed_counts(self):
        arities = {"rz": 1, "p": 1, "z": 1, "sdg": 1, "h": 1, "sx": 1, "sxdg": 1,
                   "x": 1, "cx": 2, "cz": 2, "rzz": 2, "crz": 2, "crzz": 3,
                   "rxx": 2, "ryy": 2, "xx_plus_yy": 2}
        for name, n in arities.items():
            for angle in (-.83, 0.0, .51):
                with self.subTest(gate=name, angle=angle):
                    ir = self.m.Circuit((self.m.Gate(name, tuple(reversed(range(n))), angle, -.37),), -.13)
                    qc = s.circuit_from_ir(ir, n)
                    self.connection(qc, ir, "gate:" + name)
                    self.same_unitary(qc, s.circuit_from_ir(self.m.lower(ir), n))

    def test_gaussian_runtime_order_inverse_and_unitary(self):
        from qiskit import QuantumCircuit
        from trotterlib.df_partial_s2 import QiskitDFPartialS2CircuitBuilder
        _, _, blocks, bases, _ = s.artificial_inputs()
        for block, basis in zip(blocks, bases):
            for inverse in (False, True):
                qc = QuantumCircuit(2)
                QiskitDFPartialS2CircuitBuilder._append_basis(qc, block, inverse=inverse)
                self.connection(qc, basis.circuit(inverse), "gaussian:" + str(inverse))

    def test_qubit_ordering_fermionic_minors_and_vacuum_phase(self):
        from trotterlib.df_trotter.ops import U_to_qiskit_ops_jw_givens
        _, _, _, _, matrices = s.artificial_inputs()
        # SWAP of modes must give a minus sign to the doubly occupied state.
        matrices = (*matrices, self.np.array([[0, 1], [1, 0]], complex))
        for i, matrix in enumerate(matrices):
            basis = self.m.basis_from_runtime("oracle", "artificial", U_to_qiskit_ops_jw_givens(matrix))
            g = self.Operator(s.circuit_from_ir(basis.circuit(), 2)).data
            oracle = s.exterior_fock_lift(matrix)
            vacuum = g[0, 0]
            residual = float(self.np.max(self.np.abs(g - vacuum * oracle)))
            self.assertLess(residual, 1e-13)
            self.assertAlmostEqual(abs(vacuum), 1)
            if i == 2:
                self.assertAlmostEqual((g / vacuum)[3, 3].real, -1)
            s.OBSERVATIONS.append({"label": "fermion_minors:" + str(i), "residual": residual,
                "standalone_vacuum_phase_radians": float(self.np.angle(vacuum)),
                "phase_removed_only_for_physical_fock_oracle": True})

    def test_existing_onebody_and_squared_primitive_producers(self):
        from qiskit import QuantumCircuit
        from trotterlib.df_trotter import ops
        for time in (-.3, 0, .3):
            primitives = (ops.one_body_diagonal_primitives(self.np.array([.7, -.2]), time),
                          ops.df_squared_diagonal_primitives(self.np.array([.4, -.9]), -.6, time))
            for j, p in enumerate(primitives):
                for mode in ("uncontrolled", "ordinary", "directional"):
                    qc = QuantumCircuit(3)
                    if mode == "directional": NATIVE.append_directional_diagonal(qc, p, 2)
                    else: ops.append_diagonal_primitives(qc, p, controlled=mode == "ordinary", ancilla_qubit=2)
                    self.connection(qc, self.m.diagonal_from_primitives(p, mode, 2), f"diagonal:{j}:{time}:{mode}")

    def test_explicit_zero_rz_regression_all_appenders(self):
        from qiskit import QuantumCircuit
        from trotterlib.df_trotter.ops import DiagonalEvolutionPrimitives, append_diagonal_primitives
        p = DiagonalEvolutionPrimitives(0.0, ((0, 0.0),), ((0, 1, 0.0),))
        for mode in ("uncontrolled", "ordinary", "directional"):
            qc = QuantumCircuit(3)
            if mode == "directional": NATIVE.append_directional_diagonal(qc, p, 2)
            else: append_diagonal_primitives(qc, p, controlled=mode == "ordinary", ancilla_qubit=2)
            self.connection(qc, self.m.diagonal_from_primitives(p, mode, 2), "zero-regression:" + mode)

    def test_s2_s4_ordinary_directional_scalar_wrapper(self):
        from trotterlib.df_trotter import ops
        _, _, blocks, bases, _ = s.artificial_inputs()
        def factory(j, time):
            block = blocks[j]
            return (ops.one_body_diagonal_primitives(self.np.array(block.diagonal_eigenvalues), time)
                    if j == 0 else ops.df_squared_diagonal_primitives(self.np.array(block.diagonal_eta), block.lam, time))
        for formula, policy, q in itertools.product(("2nd", "4th"), ("ordinary", "symmetric_directional"), (1, 2)):
            actual = NATIVE.build_deterministic_native(blocks, num_system_qubits=2, T=.8, q=q,
                formula=formula, scalar=.17, control_policy=policy, max_instructions=2000)
            a = self.m.deterministic_circuit(bases, factory, .8, q, formula, policy, 2, -.17 * .8)
            label = f"native:{formula}:{policy}:q{q}"
            self.connection(actual.circuit, a, label)
            b = self.m.deterministic_circuit(bases, factory, .8, q, formula, policy, 2, -.17 * .8, "B")
            residual = self.same_unitary(actual.circuit, s.circuit_from_ir(b, 3))
            self.assertLess(self.m.count(b).raw, self.m.count(a).raw)
            s.OBSERVATIONS.append({"label": label + ":B", "unitary_max_abs": residual,
                "classification": "VALID_MODEL_OPTIMIZATION_NOT_EMITTED_BY_NATIVE_PF_BUILDER",
                "A_cost": self.m.count(a).as_dict(), "B_cost": self.m.count(b).as_dict()})
            for axis, source_axis in (("X", "cosine"), ("Y", "sine")):
                wrapped = NATIVE.build_native_hadamard_wrapper(actual, axis=source_axis,
                    include_measurement=False, max_instructions=2000)
                self.connection(wrapped, self.m.hadamard_wrapper(a, 2, axis), label + ":" + axis)

    def test_scalar_only_and_measurement_scope(self):
        actual = NATIVE.build_deterministic_native((), num_system_qubits=2, T=.8, q=1,
                formula="2nd", scalar=.17, control_policy="ordinary", max_instructions=20)
        ir = self.m.Circuit((self.m.Gate("p", (2,), -.17 * .8),))
        self.connection(actual.circuit, ir, "single-native-scalar")
        for axis in ("cosine", "sine"):
            a = NATIVE.build_native_hadamard_wrapper(actual, axis=axis, include_measurement=False, max_instructions=20)
            b = NATIVE.build_native_hadamard_wrapper(actual, axis=axis, include_measurement=True, max_instructions=20)
            self.assertEqual(len(b.data), len(a.data) + 1)
            self.assertEqual(b.count_ops()["measure"], 1)

    def test_model_b_id_conflict_registry_and_binding(self):
        from qiskit.circuit.library import RZGate
        from trotterlib.df_rte_tail import DFBasisRegistry
        registry = DFBasisRegistry()
        a = registry.register(((RZGate(.2), (0,)),), num_system_qubits=2, basis_id="same")
        with self.assertRaises(ValueError):
            registry.register(((RZGate(.3), (0,)),), num_system_qubits=2, basis_id="same")
        with self.assertRaises(ValueError):
            registry.register(((RZGate(.2), (1,)),), num_system_qubits=2, basis_id="same")
        b = registry.register(((RZGate(.2), (1,)),), num_system_qubits=2, basis_id="other-qubit")
        ta, tb = [self.m.basis_from_runtime(d.basis_id, d.basis_hash, d.runtime_operations) for d in (a, b)]
        parts = [self.m.Sandwich(t, self.m.Circuit()) for t in (ta, tb)]
        self.assertEqual(self.m.count(self.m.assemble(parts, "B")).raw, 4)
        corrupted = self.m.Basis(ta.basis_id, ta.source_hash, tb.operations)
        with self.assertRaises(ValueError): self.m.assemble([parts[0], self.m.Sandwich(corrupted, self.m.Circuit())], "B")

    def test_rte_actual_builder_order_0_2_4_signed_identity_z_zz(self):
        registry, components, specs, theory = s.rte_inputs(identity=True)
        _, all_events = s.build_events(components)
        # Deliberate indices, no sampling. Includes mixed and repeated bases.
        events = [e for e in all_events if e.taylor_order == 0]
        events += [next(e for e in all_events if e.taylor_order == 2 and len(set(e.selected_component_ids)) == 3)]
        events += [next(e for e in all_events if e.taylor_order == 4 and len(set(e.selected_component_ids)) == 3)]
        def lookup(app):
            definition = registry.definition(app.basis_id)
            self.assertEqual(definition.basis_hash, app.basis_hash)
            self.assertEqual(definition.metadata.operations, app.basis_change_operations)
            return self.m.basis_from_runtime(definition.basis_id, definition.basis_hash, definition.runtime_operations)
        for event in events:
            self.assertEqual(len(event.product_component_ids), event.taylor_order)
            self.assertEqual([a.role for a in event.application_sequence], ["product"] * event.taylor_order + ["rotation"])
            for model in ("A", "B"):
                qc = s.build_rte(registry, specs, (event,), model == "B").circuit
                ir = self.m.existing_rte_occurrence((event,), lookup, 2, model)
                self.connection(qc, ir, f"rte:k{event.taylor_order}:{model}")

    def test_rte_same_different_basis_identity_and_event_boundaries(self):
        for same, identity in ((True, False), (False, False), (True, True)):
            registry, components, specs, theory = s.rte_inputs(same_basis=same, identity=identity)
            _, events = s.build_events(components, maximum_order=0)
            sequence = (events[0], events[-1], events[0])
            lookup = lambda app: self.m.basis_from_runtime(app.basis_id, app.basis_hash, registry.operations(app.basis_id))
            for model in ("A", "B"):
                actual = s.build_rte(registry, specs, sequence, model == "B")
                ir = self.m.existing_rte_occurrence(sequence, lookup, 2, model)
                self.connection(actual.circuit, ir, f"rte-boundaries:same{same}:I{identity}:{model}")

    def test_closed_occurrence_scope_differs_from_flattened_events(self):
        from qiskit import QuantumCircuit
        registry, components, specs, _ = s.rte_inputs(same_basis=True)
        _, events = s.build_events(components, maximum_order=0)
        event = events[0]
        one = s.build_rte(registry, specs, (event,), True).circuit
        closed = QuantumCircuit(3); closed.compose(one, inplace=True); closed.compose(one, inplace=True)
        flat = s.build_rte(registry, specs, (event, event), True).circuit
        self.assertGreater(len(closed.data), len(flat.data))
        self.same_unitary(closed, flat)
        s.OBSERVATIONS.append({"label": "closed-versus-flat-occurrences", "classification": "BOUNDARY_SCOPE_DIFFERENCE",
            "closed_raw": len(closed.data), "flat_raw": len(flat.data)})

    def test_general_identity_phase_zero_and_two_pi_literal(self):
        from dataclasses import replace
        from trotterlib.df_rte_circuit import DFRTEIdentityCircuitSpec
        registry, components, specs, _ = s.rte_inputs(identity=True)
        distribution, events = s.build_events(components)
        zero = next(e for e in events if e.taylor_order == 4 and all(c == "Iminus" for c in e.product_component_ids)
                    and e.rotation_component_id == "Zplus")
        # Four negative products give 4pi: source must retain the P instruction.
        actual = s.build_rte(registry, specs, (zero,), True)
        self.assertAlmostEqual(actual.relative_ancilla_phase, 4 * math.pi)
        self.assertEqual(actual.circuit.data[-1].operation.name, "p")
        self.assertEqual(s.qiskit_cost(actual.circuit).rz, self.m.count(
            self.m.existing_rte_occurrence((zero,), lambda app: self.m.basis_from_runtime(
                app.basis_id, app.basis_hash, registry.operations(app.basis_id)), 2, "B")).rz)
        # Artificial +I and -I rotations at k0 cancel their accumulated float phase.
        negative = replace(components[-1], probability=.5)
        positive = replace(negative, component_id="Iplus", coefficient_sign=1)
        _, identities = s.build_events((negative, positive), maximum_order=0)
        identity_specs = (specs[-1], DFRTEIdentityCircuitSpec("Iplus", 1.0, 1, 2))
        cancelled = s.build_rte(registry, identity_specs, identities, True)
        self.assertEqual(cancelled.relative_ancilla_phase, 0.0)
        self.assertEqual(len(cancelled.circuit.data), 0)

    def test_finite_distribution_and_component_probability_conventions(self):
        registry, components, specs, theory = s.rte_inputs(identity=True)
        distribution, events = s.build_events(components)
        model = self.m.canonical_order_probabilities(.4, 4)
        for k, p in zip(distribution.orders, distribution.order_probabilities):
            self.assertAlmostEqual(float(model[k]), p, delta=3e-16)
        for event in events:
            p_order = distribution.order_probabilities[distribution.orders.index(event.taylor_order)]
            expected = p_order * (1 / 3) ** (event.taylor_order + 1)
            self.assertAlmostEqual(event.event_probability, expected, delta=2e-16)
        self.assertAlmostEqual(math.fsum(e.event_probability for e in events), 1, delta=2e-15)

    def test_expected_cost_against_actual_qiskit_full_enumeration(self):
        from dataclasses import replace
        for same, maximum_order, repetitions, identity in ((False, 4, 1, False), (True, 2, 2, False), (True, 0, 2, True)):
            registry, components, specs, theory = s.rte_inputs(same_basis=same, identity=identity)
            distribution, events = s.build_events(components, maximum_order=maximum_order)
            probabilities = s.normalized_fractions(distribution.order_probabilities)
            orders = dict(zip(distribution.orders, probabilities))
            theoretical_by_id = {c.component_id: c for c in theory}
            def weight(event):
                return orders[event.taylor_order] * math.prod(theoretical_by_id[c].probability for c in event.selected_component_ids)
            for model in ("A", "B"):
                total = self.m.Cost()
                for sequence in itertools.product(events, repeat=repetitions):
                    actual = s.build_rte(registry, specs, sequence, model == "B").circuit
                    total += s.qiskit_cost(actual) * math.prod(weight(e) for e in sequence)
                result = self.m.expected_rte_occurrence(theory, orders, repetitions, .4, 2, model)
                for low, observed, high in zip(result.lower.values(), total.values(), result.upper.values()):
                    self.assertLessEqual(low, observed); self.assertLessEqual(observed, high)
                self.assertEqual(result.exact, not identity)
                s.OBSERVATIONS.append({"label": f"expectation:{model}:K{maximum_order}:r{repetitions}:I{identity}",
                    "qiskit_sequences_enumerated": len(events) ** repetitions, "exact": result.exact,
                    "probability_law": "NORMALIZED_FRACTIONS_OF_STORED_BINARY64_ORDER_PROBABILITIES",
                    "lower": {k: str(v) for k, v in result.lower.as_dict().items()},
                    "qiskit_expectation": {k: str(v) for k, v in total.as_dict().items()},
                    "upper": {k: str(v) for k, v in result.upper.as_dict().items()}})

    def test_native_partial_whole_trajectory_matches_closed_model(self):
        from trotterlib.df_trotter import ops
        registry, components, specs, _ = s.rte_inputs(same_basis=True)
        _, events = s.build_events(components, maximum_order=0)
        _, _, blocks, bases, _ = s.artificial_inputs()
        def factory(j, time):
            block = blocks[j]
            return ops.one_body_diagonal_primitives(self.np.array(block.diagonal_eigenvalues), time) if j == 0 else ops.df_squared_diagonal_primitives(self.np.array(block.diagonal_eta), block.lam, time)
        lookup = lambda app: self.m.basis_from_runtime(app.basis_id, app.basis_hash, registry.operations(app.basis_id))
        sequence = (events[0], events[1])
        for model in ("A", "B"):
            one = s.build_rte(registry, specs, sequence, model == "B").circuit
            actual = NATIVE.build_partial_native(blocks, num_system_qubits=2, T=.8, q=2, scalar=.17,
                controlled_tails=(one, one), control_policy="symmetric_directional", max_instructions=2000)
            prefix = self.m.partial_prefix_circuit(bases, factory, .8, 2, 2, "symmetric_directional", -.17 * .8, model)
            self.assertEqual(s.qiskit_cost(actual.circuit), self.m.count(prefix) + s.qiskit_cost(one) * 2)
            parts = []
            for _ in range(2):
                for indices, mode in ((range(2), "uncontrolled"), (reversed(range(2)), "directional")):
                    half = self.m.assemble([self.m.Sandwich(bases[j], self.m.diagonal_from_primitives(
                        factory(j, .8 / 2 / 2), mode, 2)) for j in indices], model)
                    parts.append(half)
                    if mode == "uncontrolled":
                        parts.append(self.m.existing_rte_occurrence(sequence, lookup, 2, model))
            parts.append(self.m.Circuit((self.m.Gate("p", (2,), -.17 * .8),)))
            self.connection(actual.circuit, self.m.assemble(parts, model), "native-partial-closed:" + model)
            self.assertEqual(actual.circuit.data[-1].operation.name, "p")


if __name__ == "__main__":
    unittest.main()
