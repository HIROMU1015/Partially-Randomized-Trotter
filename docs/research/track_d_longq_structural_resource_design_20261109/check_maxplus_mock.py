"""Integer-only design checks; no circuits, P0, scientific payloads or simulator.

Reuse the existing C7-2 weighted_depth function verbatim via its AST, without
importing that module's Qiskit/numerical dependencies. The small matrix helpers
below are a mathematical mock of the proposed interface, not a price API.
"""
import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as S
import unittest


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src/trottertracks/track_d_c7_2_toy.py"
tree = ast.parse(SOURCE.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "weighted_depth")
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
weighted_depth = namespace["weighted_depth"]


def legacy(word, qubits, clbits=0, metric=None):
    """Adapt only synthetic tuples to the existing function's circuit protocol."""
    circuit = S(qubits=tuple(range(qubits)), clbits=tuple(range(qubits, qubits+clbits)),
                data=[S(operation=S(name=name), qubits=tuple(i for i in wires if i < qubits),
                        clbits=tuple(i for i in wires if i >= qubits)) for name, wires in word])
    result = weighted_depth(circuit, metric)
    return tuple(result["qubit_finish"] + result["classical_finish"])


def transfer(word, width, metric=None):
    """None is minus infinity; no floating-point operations are needed."""
    rows = [[0 if i == j else None for j in range(width)] for i in range(width)]
    for name, wires in word:
        weight = int(metric is None or name == metric)
        combined = []
        for j in range(width):
            values = [rows[i][j] for i in wires if rows[i][j] is not None]
            combined.append(max(values) + weight if values else None)
        for i in wires:
            rows[i] = list(combined)
    return rows


def apply(matrix, arrival):
    return tuple(max(a+b for a, b in zip(row, arrival) if a is not None) for row in matrix)


def compose(later, earlier):
    width = len(later)
    out = []
    for i in range(width):
        row = []
        for j in range(width):
            terms = [later[i][k] + earlier[k][j] for k in range(width)
                     if later[i][k] is not None and earlier[k][j] is not None]
            row.append(max(terms) if terms else None)
        out.append(row)
    return out


class MaxPlusMock(unittest.TestCase):
    def test_existing_counter_for_three_metrics_and_measurement(self):
        words = [[], [("rz", (0,))],
                 [("rz", (0,)), ("rz", (0,)), ("cx", (0, 1)), ("rz", (1,))],
                 [("x", (0,)), ("rz", (1,)), ("cx", (0, 1)), ("rz", (0,)),
                  ("cx", (1, 2)), ("measure", (0, 3))]]
        for word in words:
            for metric in (None, "rz", "cx"):
                with self.subTest(word=word, metric=metric):
                    self.assertEqual(apply(transfer(word, 4, metric), (0,)*4),
                                     legacy(word, 3, 1, metric))

    def test_zero_input_exit_vector_is_insufficient(self):
        independent = transfer([("x", (0,)), ("x", (1,))], 2)
        coupled = transfer([("cx", (0, 1))], 2)
        self.assertEqual(apply(independent, (0, 0)), apply(coupled, (0, 0)))
        self.assertEqual(apply(independent, (10, 0)), (11, 1))
        self.assertEqual(apply(coupled, (10, 0)), (11, 11))

    def test_composition_has_chronological_order(self):
        a, b = [("rz", (0,)), ("rz", (0,))], [("cx", (0, 1))]
        for metric in (None, "rz", "cx"):
            joined = compose(transfer(b, 2, metric), transfer(a, 2, metric))
            self.assertEqual(joined, transfer(a+b, 2, metric))
            self.assertEqual(apply(joined, (7, 2)), apply(transfer(a+b, 2, metric), (7, 2)))
        self.assertNotEqual(compose(transfer(a, 2), transfer(b, 2)), transfer(a+b, 2))

    def test_excluded_cx_still_synchronizes_rz_arrivals(self):
        word = [("rz", (0,)), ("rz", (0,)), ("cx", (0, 1)), ("rz", (1,))]
        self.assertEqual(max(apply(transfer(word, 2, "rz"), (0, 0))), 3)
        deleted = [op for op in word if op[0] != "cx"]
        self.assertEqual(max(apply(transfer(deleted, 2, "rz"), (0, 0))), 2)

    def test_chunk_boundary_keeps_dependencies_and_counts(self):
        word = [("x", (0,)), ("cx", (0, 1)), ("rz", (1,)), ("cx", (1, 2))]
        for cut in range(len(word)+1):
            left, right = word[:cut], word[cut:]
            self.assertEqual(compose(transfer(right, 3), transfer(left, 3)), transfer(word, 3))
            self.assertEqual(Counter(n for n, _ in left) + Counter(n for n, _ in right),
                             Counter(n for n, _ in word))
        # These are counts of the fixed word; no P0 boundary simplification is modeled.

    def test_shared_ancilla_shell_keeps_wire_mapping(self):
        # Abstract three-wire dependencies with unit weight, NOT native CSWAP prices.
        shell = [("abstract_shared", (4, 0, 2)), ("abstract_shared", (4, 1, 3))]
        lane_a = [("x", (0,))]*4 + [("rz", (1,))]
        lane_b = [("rz", (2,)), ("x", (3,)), ("x", (3,))]
        whole = shell + lane_a + lane_b + shell
        lanes_ab = transfer(lane_a+lane_b, 5)
        lanes_ba = transfer(lane_b+lane_a, 5)
        self.assertEqual(lanes_ab, lanes_ba)
        matrix = compose(transfer(shell, 5), compose(lanes_ab, transfer(shell, 5)))
        self.assertEqual(apply(matrix, (0,)*5), legacy(whole, 5))
        self.assertEqual(apply(matrix, (9, 0, 3, 0, 1)),
                         apply(transfer(whole, 5), (9, 0, 3, 0, 1)))

    def test_empty_template_is_identity(self):
        identity = transfer([], 3)
        self.assertEqual(apply(identity, (4, 0, 11)), (4, 0, 11))
        other = transfer([("cx", (0, 2))], 3)
        self.assertEqual(compose(identity, other), other)
        self.assertEqual(compose(other, identity), other)


if __name__ == "__main__":
    unittest.main(verbosity=2)
