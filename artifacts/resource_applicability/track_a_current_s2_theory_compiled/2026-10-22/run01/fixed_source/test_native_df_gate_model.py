"""Artificial algebra/counting checks only; no scientific dependencies or data."""
import cmath
from fractions import Fraction as F
import importlib.util
import itertools
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_df_gate_model", ROOT / "src/trotterlib/native_df_gate_model.py")
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


def direct_matrix(g):
    """Independent analytic matrix oracle, little-endian local qubit order."""
    n, a = len(g.qubits), g.angle
    dim = 1 << n
    out = [[complex(i == j) for j in range(dim)] for i in range(dim)]
    if g.name in ("rz", "p", "sdg", "z", "rzz", "crz", "crzz"):
        for j in range(dim):
            if g.name == "rz":
                phi = -a / 2 * (-1) ** (j & 1)
            elif g.name in ("p", "sdg", "z"):
                phi = (a if g.name == "p" else -math.pi / 2 if g.name == "sdg" else math.pi) * j
            elif g.name == "rzz":
                phi = -a / 2 * (-1) ** j.bit_count()
            else:
                phi = -a / 2 * (-1) ** (j >> 1).bit_count() if j & 1 else 0
            out[j][j] = cmath.exp(1j * phi)
        return out
    if g.name == "h":
        return [[1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]]
    if g.name in ("sx", "sxdg"):
        z = 1j if g.name == "sx" else -1j
        return [[(1 + z) / 2, (1 - z) / 2], [(1 - z) / 2, (1 + z) / 2]]
    if g.name in ("x", "cx"):
        return [[complex(i == (j ^ (1 if n == 1 else 2 if j & 1 else 0)))
                 for j in range(dim)] for i in range(dim)]
    if g.name == "cz":
        out[3][3] = -1
        return out
    if g.name in ("rxx", "ryy"):
        out = [[complex(i == j) * math.cos(a / 2) for j in range(4)] for i in range(4)]
        for j in range(4):
            eigen = 1 if g.name == "rxx" else -(-1) ** j.bit_count()
            out[j ^ 3][j] = -1j * math.sin(a / 2) * eigen
        return out
    if g.name == "xx_plus_yy":
        out[1][1] = out[2][2] = math.cos(a / 2)
        out[1][2] = -1j * math.sin(a / 2) * cmath.exp(-1j * g.beta)
        out[2][1] = -1j * math.sin(a / 2) * cmath.exp(1j * g.beta)
        return out
    raise AssertionError(g.name)


def unitary(circuit, n):
    columns = []
    for start in range(1 << n):
        vector = [complex(j == start) for j in range(1 << n)]
        for gate in circuit.operations:
            matrix = direct_matrix(gate)
            after = [0j] * len(vector)
            mask = sum(1 << q for q in gate.qubits)
            for j, amp in enumerate(vector):
                local = sum(((j >> q) & 1) << p for p, q in enumerate(gate.qubits))
                for i in range(len(matrix)):
                    target = (j & ~mask) | sum(((i >> p) & 1) << q for p, q in enumerate(gate.qubits))
                    after[target] += matrix[i][local] * amp
            vector = after
        columns.append([amp * cmath.exp(1j * circuit.global_phase) for amp in vector])
    return columns


def basis(key="a", qubit=0, angle=0.37):
    return m.Basis(key, "synthetic-" + key, (m.Gate("h", (qubit,)), m.Gate("rz", (qubit,), angle)))


def event_patterns(components, orders):
    patterns = []
    for k, p in orders.items():
        for sequence in itertools.product(components, repeat=k + 1):
            probability = p * math.prod(c.probability for c in sequence)
            patterns.append(((k, sequence[:-1], sequence[-1]), probability))
    return patterns


class MatrixChecks(unittest.TestCase):
    def assertSame(self, a, b, n):
        left, right = unitary(a, n), unitary(b, n)
        self.assertLess(max(abs(x - y) for u, v in zip(left, right) for x, y in zip(u, v)), 4e-14)

    def test_declared_lowering_all_gates_including_global_phase(self):
        arities = {"p": 1, "z": 1, "sdg": 1, "h": 1, "sxdg": 1,
                   "cz": 2, "rzz": 2, "crz": 2, "crzz": 3,
                   "rxx": 2, "ryy": 2, "xx_plus_yy": 2}
        for name, n in arities.items():
            for angle in (-1.17, 0.0, 0.83):
                with self.subTest(name=name, angle=angle):
                    g = m.Gate(name, tuple(range(n)), angle, -0.61)
                    self.assertSame(m.Circuit((g,)), m.lower(m.Circuit((g,))), n)

    def test_gaussian_forward_inverse_exact_relative_phase(self):
        b = m.Basis("two", "fixture", (m.Gate("rz", (1,), 0.21),
                    m.Gate("xx_plus_yy", (1, 0), 0.7, -0.4)))
        self.assertSame(m.Circuit((*b.circuit(True).operations, *b.operations)), m.Circuit(), 2)
        self.assertSame(b.circuit(), m.lower(b.circuit()), 2)

    def test_basis_cancellation_preserves_noncommuting_order(self):
        b, c = basis(), basis("b", 0, -0.4)
        parts = [m.Sandwich(b, m.Circuit((m.Gate("crz", (1, 0), 0.3),))),
                 m.Sandwich(b, m.Circuit((m.Gate("cz", (1, 0)),))),
                 m.Sandwich(c, m.Circuit((m.Gate("crz", (1, 0), -0.8),)))]
        a, optimized = m.assemble(parts), m.assemble(parts, "B")
        self.assertSame(a, optimized, 2)
        self.assertEqual(m.count(a).raw - m.count(optimized).raw, 4)

    def test_symmetric_directional_s2_equals_ordinary_control(self):
        bases = (basis(), basis("b", 0, -0.4))
        def factory(j, t):
            return NS(rz=((0, (j + 1) * t),), rzz=(), global_phase=-0.23 * t)
        for formula in ("2nd", "4th"):
            ordinary = m.deterministic_circuit(bases, factory, 0.7, 2, formula, ancilla=1, scalar_phase=-0.13)
            directional = m.deterministic_circuit(bases, factory, 0.7, 2, formula,
                                                  "symmetric_directional", 1, -0.13, "B")
            self.assertSame(ordinary, directional, 2)

    def test_identity_negative_product_phase_is_not_discarded(self):
        identity = m.Component("Iminus", F(1), None, sign=-1)
        circuit = m.explicit_rte_occurrence(((2, (identity, identity), identity),), .4, 0)
        phase = math.pi + math.atan(.4 / 3)
        self.assertSame(circuit, m.Circuit((m.Gate("p", (0,), phase),)), 1)
        self.assertEqual(len(circuit.operations), 1)

    def test_artificial_df_squared_occupation_eigenvalues(self):
        # eta=(1,2), lambda=.4, time=.3, calculated by hand from source formula.
        p = NS(rz=((0, -.36), (1, -.72)), rzz=((0, 1, .24),), global_phase=-.42)
        u = unitary(m.diagonal_from_primitives(p, "uncontrolled"), 2)
        for j in range(4):
            occupation_sum = (j & 1) + 2 * ((j >> 1) & 1)
            self.assertLess(abs(u[j][j] - cmath.exp(-1j * .12 * occupation_sum**2)), 1e-14)
        # The ancilla-relative global phase must also survive control.
        controlled = unitary(m.diagonal_from_primitives(p, "ordinary", 2), 3)
        for j in range(8):
            expected = u[j & 3][j & 3] if j & 4 else 1
            self.assertLess(abs(controlled[j][j] - expected), 1e-14)

    def test_onebody_occupation_eigenvalues(self):
        p = NS(rz=((0, -.21), (1, .12)), rzz=(), global_phase=(-.21 + .12) / 2)
        u = unitary(m.diagonal_from_primitives(p, "uncontrolled"), 2)
        for j in range(4):
            self.assertLess(abs(u[j][j] - cmath.exp(1j * (-.21 * (j & 1) + .12 * ((j >> 1) & 1)))), 1e-14)


class CountChecks(unittest.TestCase):
    def test_profile_closed_counts(self):
        expected = {"h": (2, 0, 1, 0), "cz": (4, 1, 2, 0),
                    "crz": (2, 2, 0, 0), "crzz": (2, 4, 0, 0),
                    "rxx": (9, 2, 4, 0), "ryy": (1, 2, 4, 2),
                    "xx_plus_yy": (12, 4, 8, 2)}
        for name, counts in expected.items():
            n = 3 if name == "crzz" else 1 if name == "h" else 2
            actual = m.count(m.Circuit((m.Gate(name, tuple(range(n)), .4, .7),)))
            self.assertEqual(actual.values(), (1, *counts))

    def test_diagonal_modes_zero_rules_and_scalar(self):
        p = NS(rz=((0, .2), (1, 0.0)), rzz=((0, 1, -.7),), global_phase=.31)
        self.assertEqual(m.count(m.diagonal_from_primitives(p, "uncontrolled")), m.Cost(3, 3, 2))
        self.assertEqual(m.count(m.diagonal_from_primitives(p, "ordinary", 2)), m.Cost(4, 7, 8))
        self.assertEqual(m.count(m.diagonal_from_primitives(p, "directional", 2)), m.Cost(12, 4, 8))
        self.assertEqual(m.diagonal_from_primitives(p, "uncontrolled").global_phase, .31)

    def test_wrapper_axes(self):
        self.assertEqual(m.count(m.hadamard_wrapper(m.Circuit(), 2, "X")), m.Cost(2, 4, 0, 2))
        self.assertEqual(m.count(m.hadamard_wrapper(m.Circuit(), 2, "Y")), m.Cost(3, 5, 0, 2))

    def test_literal_zero_gaussian_is_counted(self):
        self.assertEqual(m.count(m.Circuit((m.Gate("rz", (0,), 0.0),))), m.Cost(1, 1))

    def test_structural_bound_is_not_lowered_gate_count(self):
        p = NS(rz=((0, .2), (1, .3)), rzz=((0, 1, -.7),), global_phase=.31)
        for mode in ("uncontrolled", "ordinary", "directional"):
            word = m.assemble((m.Sandwich(basis(), m.diagonal_from_primitives(p, mode, 2)),))
            self.assertLessEqual(m.count(word).raw, m.structural_instruction_bound(2, 2, mode=mode))
        self.assertEqual(m.structural_instruction_bound(2, 2, mode="directional"), 16)

    def test_pf_schedule_and_negative_s4_middle(self):
        schedule = m.pf_schedule(2, .8, 2, "4th", "symmetric_directional")
        self.assertEqual(len(schedule), 24)
        self.assertEqual([s.block for s in schedule[:4]], [0, 1, 1, 0])
        self.assertLess(schedule[4].time, 0)
        self.assertEqual([s.mode for s in schedule[:4]], ["uncontrolled"] * 2 + ["directional"] * 2)
        for j in (0, 1):
            self.assertAlmostEqual(sum(s.time for s in schedule if s.block == j), .8)

    def test_partial_prefix_boundaries_are_closed(self):
        b = basis()
        factory = lambda j, t: NS(rz=((0, t),), rzz=(), global_phase=0)
        prefix = m.partial_prefix_circuit((b,), factory, .8, 2, 1, model="B")
        # Four separate halves; no cancellation is credited through an omitted tail.
        self.assertEqual(m.count(prefix).raw, 4 * (4 + 1))

    def test_raw_histogram(self):
        c = m.Circuit((m.Gate("h", (0,)), m.Gate("crz", (1, 0), .2)))
        self.assertEqual(m.primitive_histogram(c), {"crz": 1, "h": 1})

    def test_adapter_uses_runtime_metadata(self):
        b = m.basis_from_runtime("a", "hash", ((NS(name="rz", params=[.3]), (0,)),))
        self.assertEqual(b.operations, (m.Gate("rz", (0,), .3),))
        with self.assertRaises(ValueError):
            m.basis_from_runtime("a", "hash", ((NS(name="unitary", params=[]), (0, 1)),))

    def test_existing_event_adapter_and_order_rejection(self):
        b = basis()
        apps = [NS(application_index=i, component_id="a", role=role, is_identity=False,
                   basis_id=b.basis_id, basis_hash=b.source_hash,
                   diagonal_pauli_support=(0,), coefficient_sign=-1)
                for i, role in enumerate(("product", "product", "rotation"))]
        event = NS(taylor_order=2, application_sequence=apps, product_component_ids=("a", "a"),
                   rotation_component_id="a", phase=-1, rotation_angle=math.atan(.4 / 3))
        comp = m.Component("a", F(1), b, (0,), -1)
        original = m.explicit_rte_occurrence(((2, (comp, comp), comp),), .4, 2, "B")
        adapted = m.existing_rte_occurrence((event,), lambda app: b, 2, "B")
        self.assertEqual(adapted, original)
        apps[0].role = "rotation"
        with self.assertRaises(ValueError):
            m.existing_rte_occurrence((event,), lambda app: b, 2)

    def test_identity_validation(self):
        b = basis()
        other = m.Basis(b.basis_id, b.source_hash, (m.Gate("h", (1,)),))
        with self.assertRaises(ValueError):
            m.assemble((m.Sandwich(b, m.Circuit()), m.Sandwich(other, m.Circuit())), "B")
        with self.assertRaises(ValueError):
            m.Basis("controlled", "x", (), "controlled")
        with self.assertRaises(ValueError):
            m.diagonal_from_primitives(NS(rz=((0, .1),), rzz=(), global_phase=0), "ordinary", 0)

    def test_identity_barrier_flushes_even_with_zero_phase(self):
        b = basis()
        parts = [m.Sandwich(b, m.Circuit()), m.Barrier(), m.Sandwich(b, m.Circuit())]
        self.assertEqual(m.count(m.assemble(parts, "B")).raw, 8)


class ExpectationChecks(unittest.TestCase):
    def compare_enumeration(self, components, orders, r, model):
        patterns = event_patterns(components, orders)
        expectation = m.Cost()
        for draws in itertools.product(patterns, repeat=r):
            cost = m.count(m.explicit_rte_occurrence([d[0] for d in draws], .4, 2, model))
            expectation += cost * math.prod(d[1] for d in draws)
        result = m.expected_rte_occurrence(components, orders, r, .4, 2, model)
        for lo, actual, hi in zip(result.lower.values(), expectation.values(), result.upper.values()):
            self.assertLessEqual(lo, actual)
            self.assertLessEqual(actual, hi)
        return result

    def test_dp_all_orders_0_2_4_and_signed_components(self):
        components = (m.Component("a", F(2, 3), basis(), (0,)),
                      m.Component("b", F(1, 3), basis("b", 1), (0, 1), -1))
        for model in ("A", "B"):
            result = self.compare_enumeration(components, {0: F(1, 2), 2: F(1, 3), 4: F(1, 6)}, 1, model)
            self.assertTrue(result.exact)

    def test_dp_cross_event_same_basis_and_different_basis(self):
        for b in (basis(), basis("b", 1)):
            components = (m.Component("a", F(1, 2), basis(), (0,)),
                          m.Component("b", F(1, 2), b, (1,)))
            for model in ("A", "B"):
                result = self.compare_enumeration(components, {0: F(3, 4), 2: F(1, 4)}, 2, model)
                self.assertTrue(result.exact)

    def test_identity_phase_is_explicit_bound(self):
        components = (m.Component("a", F(1, 2), basis(), (0,)),
                      m.Component("I", F(1, 2), None, sign=-1))
        result = self.compare_enumeration(components, {0: F(1, 2), 2: F(1, 2)}, 2, "B")
        self.assertFalse(result.exact)
        self.assertEqual(result.upper.cx, result.lower.cx)
        self.assertEqual(result.upper.rz - result.lower.rz, 1)

    def test_expected_basis_runs_closed_formula_k0(self):
        b = basis()
        components = (m.Component("a", F(1), b, (0,)),)
        a = m.expected_rte_occurrence(components, {0: F(1)}, 7, .4, 2, "A")
        c = m.expected_rte_occurrence(components, {0: F(1)}, 7, .4, 2, "B")
        saving = (m.count(b.circuit()) + m.count(b.circuit(True))) * 6
        self.assertEqual(a.lower, c.lower + saving)

    def test_rare_order_not_discarded(self):
        law = {0: 1 - F(1, 10**18), 4: F(1, 10**18)}
        result = m.expected_rte_occurrence((m.Component("a", F(1), basis(), (0,)),), law, 1, .4, 2)
        self.assertGreater(result.lower.cx, 2)
        self.assertEqual(result.lower.cx, 2 + F(4, 10**18))

    def test_canonical_weights_and_tau_zero(self):
        law = m.canonical_order_probabilities(.3, 4)
        self.assertEqual(sum(law.values()), 1)
        for k in (0, 2, 4):
            self.assertGreater(law[k], 0)
        ratio = float(law[2] / law[0])
        self.assertAlmostEqual(ratio, .3**2 / 2 * math.hypot(1, .3 / 3) / math.hypot(1, .3))
        self.assertEqual(m.canonical_order_probabilities(0, 4), {0: F(1), 2: F(0), 4: F(0)})
        with self.assertRaises(ArithmeticError):
            m.canonical_order_probabilities(1e-300, 4)

    def test_whole_trajectory_q_r_and_wrapper(self):
        occurrence = m.expected_rte_occurrence((m.Component("a", F(1), basis(), (0,)),),
                                             {0: F(1)}, 3, .4, 2, "B")
        result = m.expected_partial_trajectory(occurrence, 4, m.Circuit(), 2, "Y")
        self.assertEqual(result.lower, occurrence.lower * 4 + m.Cost(3, 5, 0, 2))

    def test_invalid_laws_and_caps_fail_closed(self):
        c = m.Component("a", F(1), basis(), (0,))
        with self.assertRaises(ValueError):
            m.expected_rte_occurrence((c,), {1: F(1)}, 1, .2, 2)
        with self.assertRaises(ValueError):
            m.expected_rte_occurrence((c,), {0: F(1, 2)}, 1, .2, 2)
        with self.assertRaises(RuntimeError):
            m.expected_rte_occurrence((c,), {4: F(1)}, 2, .2, 2, "B", max_transitions=1)


if __name__ == "__main__":
    unittest.main()
