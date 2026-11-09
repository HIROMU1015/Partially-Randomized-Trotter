"""T0 synthetic native words only: no molecular input, P0, fit or simulation."""
import ast
from collections import Counter
import importlib.util
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trottertracks import track_d_native_resources as m


def existing_function(path, name, namespace=None, *, class_name=None):
    """Reuse a frozen pure function/test without importing its scientific runner."""
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body if class_name is None else next(
        n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    function = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {} if namespace is None else dict(namespace)
    exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), scope)
    return scope[name]


legacy_depth = existing_function("src/trottertracks/track_d_c7_2_toy.py", "weighted_depth")
legacy_metrics = existing_function("src/trotterlib/track_d_minimal_reference.py", "metrics")


def rz(q, angle=.17):
    return m.NativeInstruction("rz", (q,), params=(angle,))


def x(q):
    return m.NativeInstruction("x", (q,))


def cx(a, b):
    return m.NativeInstruction("cx", (a, b))


def measure(q, c):
    return m.NativeInstruction("measure", (q,), (c,))


def arrivals(width):
    return {key: tuple((i+1)*(k+2) for i in range(width)) for k, key in enumerate(m.DEPTHS)}


def legacy(word, nq, nc=0):
    circuit = S(qubits=tuple(range(nq)), clbits=tuple(range(nq, nq+nc)),
        data=[S(operation=S(name=op.name), qubits=op.qubits,
                clbits=tuple(nq+c for c in op.clbits)) for op in word])
    return {key: tuple(row for group in (value["qubit_finish"], value["classical_finish"]) for row in group)
            for key, gate in zip(m.DEPTHS, ("rz", "cx", None))
            for value in [legacy_depth(circuit, gate)]}


class NativeTests(unittest.TestCase):
    def test_count_size_phase_and_declared_idle_width(self):
        word = [rz(0, 0.), m.NativeInstruction("sx", (1,)), x(1), cx(0, 1), measure(1, 0)]
        result = m.count_native(iter(word), 3, 1, global_phase=.37)
        self.assertEqual(result["metrics"], dict(rz_count=1, cx_count=1, rz_depth=1, cx_depth=1,
            total_depth=4, circuit_size=5, qubit_count=3, measurement_count=1, global_phase=.37))
        self.assertEqual(result["logical_width"], 3)
        self.assertEqual(result["classical_width"], 1)
        self.assertEqual(result["qualification"], "NATIVE_EXACT")
        self.assertEqual(result["p0_qualification"], "P0_UNQUALIFIED")

    def test_serial_and_parallel_lanes(self):
        word = [rz(0), rz(0), x(1), x(1)]
        self.assertEqual(m.count_native(iter(word), 2)["metrics"]["total_depth"], 2)
        self.assertEqual(m.count_native(iter(word+[cx(0, 1), x(0)]), 2)["metrics"]["total_depth"], 4)

    def test_filtered_sync_and_classical_measure_dependency(self):
        word = [rz(0), rz(0), cx(0, 1), measure(1, 0), measure(2, 0), rz(2)]
        result = m.count_native(iter(word), 3, 1)
        self.assertEqual(result["wire_arrivals"], legacy(word, 3, 1))
        self.assertEqual(result["metrics"]["rz_depth"], 3)
        deleted = [op for op in word if op.name == "rz"]
        self.assertEqual(m.count_native(iter(deleted), 3, 1)["metrics"]["rz_depth"], 2)

    def test_nonuniform_arrivals_and_integer_precision(self):
        word = [rz(0), cx(0, 1), measure(1, 0)]
        start = {key: (10**24, 0, 3) for key in m.DEPTHS}
        direct = m.count_native(iter(word), 2, 1, arrivals=start)
        template = m.NativeTemplate.from_stream(iter(word), 2, 1)
        self.assertEqual(template.result(start), direct)
        self.assertEqual(direct["metrics"]["total_depth"], 10**24+3)
        self.assertEqual(direct["metrics"]["rz_depth"], 10**24+1)
        self.assertEqual(direct["metrics"]["cx_depth"], 10**24+1)

    def test_zero_input_exit_vector_is_not_a_transfer(self):
        separate = m.NativeTemplate.from_stream(iter([x(0), x(1)]), 2)
        linked = m.NativeTemplate.from_stream(iter([cx(0, 1)]), 2)
        self.assertEqual(separate.apply()["total_depth"], linked.apply()["total_depth"])
        start = {key: (10, 0) for key in m.DEPTHS}
        self.assertEqual(separate.apply(start)["total_depth"], (11, 1))
        self.assertEqual(linked.apply(start)["total_depth"], (11, 11))

    def test_one_pass_generator_with_backpressure(self):
        counter = m.NativeCounter(1)
        def source():
            for i in range(500):
                yield x(0)
                # A consumer that first materializes the stream fails here.
                self.assertEqual(counter.result()["metrics"]["circuit_size"], i+1)
        counter.extend(source())
        self.assertEqual(counter.result()["metrics"]["total_depth"], 500)

    def test_all_chunk_splits_of_small_native_words(self):
        alphabet = (rz(0), x(1), cx(0, 1), measure(1, 0))
        for length in range(4):
            for word in product(alphabet, repeat=length):
                whole = m.NativeTemplate.from_stream(iter(word), 2, 1)
                self.assertEqual(whole.apply(), legacy(word, 2, 1))
                self.assertEqual(whole.result(), m.count_native(iter(word), 2, 1))
                for cut in range(length+1):
                    with self.subTest(length=length, cut=cut, names=[o.name for o in word]):
                        left = m.NativeTemplate.from_stream(iter(word[:cut]), 2, 1)
                        right = m.NativeTemplate.from_stream(iter(word[cut:]), 2, 1)
                        combined = left.then(right)
                        self.assertEqual(combined, whole)
                        self.assertEqual(combined.result(arrivals(3)),
                                         m.count_native(iter(word), 2, 1, arrivals=arrivals(3)))

    def test_composition_order_and_no_boundary_optimization(self):
        a = m.NativeTemplate.from_stream(iter([rz(0), rz(0)]), 2)
        b = m.NativeTemplate.from_stream(iter([cx(0, 1)]), 2)
        self.assertNotEqual(a.then(b).apply(), b.then(a).apply())
        cx_pair = b.then(b).result()["metrics"]
        self.assertEqual((cx_pair["cx_count"], cx_pair["circuit_size"]), (2, 2))

    def test_empty_identity_and_no_wire_template(self):
        identity = m.NativeTemplate.from_stream(iter(()), 2, 1)
        other = m.NativeTemplate.from_stream(iter([cx(0, 1), measure(0, 0)]), 2, 1)
        self.assertEqual(identity.then(other), other)
        self.assertEqual(other.then(identity), other)
        self.assertEqual(identity.apply(arrivals(3)), arrivals(3))
        zero = m.NativeTemplate.from_stream(iter(()), 0)
        self.assertEqual(zero.then(zero), zero)
        self.assertEqual(zero.result()["metrics"]["total_depth"], 0)

    def test_independent_register_embedding_and_shared_ancilla(self):
        a_word, b_word = [rz(0), rz(0), x(1)], [x(0), rz(1)]
        a = m.NativeTemplate.from_stream(iter(a_word), 2).embed(5, qubit_map=(0, 1))
        b = m.NativeTemplate.from_stream(iter(b_word), 2).embed(5, qubit_map=(2, 3))
        self.assertEqual(a.then(b), b.then(a))
        shell_word = [cx(4, 0), cx(4, 2), cx(4, 1), cx(4, 3)]
        shell = m.NativeTemplate.from_stream(iter(shell_word), 5)
        total = shell.then(a).then(b).then(shell)
        flattened = shell_word + list(m.remap_native(iter(a_word), qubit_map=(0, 1)))
        flattened += list(m.remap_native(iter(b_word), qubit_map=(2, 3))) + shell_word
        self.assertEqual(total.result(arrivals(5)), m.count_native(iter(flattened), 5, arrivals=arrivals(5)))
        self.assertEqual(total.apply(), legacy(flattened, 5))

    def test_classical_embedding_and_wire_permutation(self):
        word = [rz(0), cx(0, 1), measure(1, 0)]
        template = m.NativeTemplate.from_stream(iter(word), 2, 1)
        mapped = template.embed(4, 2, qubit_map=(3, 1), clbit_map=(1,))
        direct = m.count_native(m.remap_native(iter(word), qubit_map=(3, 1), clbit_map=(1,)),
                                4, 2, arrivals=arrivals(6))
        self.assertEqual(mapped.result(arrivals(6)), direct)

    def test_explicit_actual_inverse_native_word(self):
        forward = [rz(0, .31), m.NativeInstruction("sx", (1,)), cx(0, 1), rz(1, -.7)]
        # Supplied actual inverse fixture, not a matrix inverse or auto-lowering:
        # SX inverse is three SX instructions in this chosen native word.
        inverse = [rz(1, .7), cx(0, 1)] + [m.NativeInstruction("sx", (1,))]*3 + [rz(0, -.31)]
        template = m.NativeTemplate.from_stream(iter(inverse), 2)
        self.assertEqual(template.apply(), legacy(inverse, 2))
        self.assertEqual(template.result(arrivals(2)), m.count_native(iter(inverse), 2, arrivals=arrivals(2)))
        self.assertEqual(template.result()["metrics"]["circuit_size"], len(forward)+2)
        self.assertFalse(hasattr(template, "inverse"))

    def test_capture_is_universal_and_snapshots_are_immutable(self):
        counter = m.NativeCounter(2, arrivals=arrivals(2), capture_transfer=True)
        counter.feed(rz(0)); before = counter.template()
        counter.feed(cx(0, 1))
        self.assertEqual(counter.template().result(arrivals(2)), counter.result())
        self.assertEqual(before.result()["metrics"]["circuit_size"], 1)
        with self.assertRaises(ValueError):m.NativeCounter(2).template()

    def test_global_phase_is_metadata_not_gate_cost(self):
        a = m.NativeTemplate.from_stream(iter([x(0)]), 1, global_phase=.125)
        b = m.NativeTemplate.from_stream(iter(()), 1, global_phase=.25)
        self.assertEqual(a.then(b).global_phase, .375)
        self.assertEqual(a.then(b).result()["metrics"]["circuit_size"], 1)

    def test_unsupported_gates_and_unknown_conditions(self):
        for name in ("crz", "rzz", "xx_plus_yy", "cswap", "h", "p", "sxdg", "id", "global_phase"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                m.NativeInstruction(name, (0,))
        for condition in (False, True, (0, 1), "unknown"):
            with self.assertRaises(ValueError):m.NativeInstruction("x", (0,), condition=condition)

    def test_invalid_native_shapes_bounds_and_parameters(self):
        for args in (("cx", (0, 0)), ("measure", (0,)), ("rz", (0,)), ("x", (True,)),
                     ("x", (-1,)), ("x", (0,), (0,))):
            with self.assertRaises(ValueError):m.NativeInstruction(*args)
        for value in (True, float("nan"), float("inf"), 1j, "symbolic"):
            with self.assertRaises(ValueError):rz(0, value)
        for op in (x(2), measure(1, 1)):
            with self.assertRaises(ValueError):m.count_native(iter([op]), 2, 1)
        with self.assertRaises(TypeError):m.count_native(iter([("x", (0,))]), 2)
        with self.assertRaises(ValueError):m.NativeCounter(True)
        with self.assertRaises(ValueError):m.NativeCounter(2, global_phase=float("nan"))

    def test_invalid_arrivals_and_maps(self):
        bad = ({}, {key:(0,) for key in m.DEPTHS}, {key:(0, -.1) for key in m.DEPTHS})
        for start in bad:
            with self.assertRaises(ValueError):m.NativeCounter(2, arrivals=start)
        template = m.NativeTemplate.from_stream(iter([cx(0, 1)]), 2)
        for mapping in ((0, 0), (0,), (0, 3)):
            with self.assertRaises(ValueError):template.embed(3, qubit_map=mapping)
        with self.assertRaises(ValueError):list(m.remap_native(iter([x(1)]), qubit_map=(0,)))
        with self.assertRaises(ValueError):list(m.remap_native(iter(()), qubit_map=(0, 0)))
        with self.assertRaises(ValueError):template.then(template.embed(3, qubit_map=(0, 1)))
        with self.assertRaises(ValueError):m.NativeTemplate(1, 0, (((None,),),)*3, ())

    def test_reuses_existing_c7_4_feature_regression(self):
        feature = existing_function("src/trottertracks/track_d_c7_4_models.py", "event_features", {"Counter":Counter})
        test = existing_function("tests/check_track_d_c7_4.py", "test_feature_max_is_pathwise",
                                 {"m":S(event_features=feature)}, class_name="Models")
        test(self)


@unittest.skipUnless(importlib.util.find_spec("qiskit"), "optional Qiskit1.3 reader/oracle tests")
class QiskitNativeTests(unittest.TestCase):
    def test_existing_metrics_oracle_and_native_reader(self):
        import qiskit
        from qiskit import QuantumCircuit
        self.assertEqual(qiskit.__version__, "1.3.0")
        for length in range(6):
            qc = QuantumCircuit(3, 2); qc.global_phase=.37
            for i in range(length):
                qc.rz(.17 if i % 2 else 0., i % 3); qc.sx((i+1) % 3)
                qc.x(i % 3); qc.cx(i % 3, (i+1) % 3)
            qc.measure(1, 0); qc.measure(2, 0)
            # A new transpile or numerical check is not part of this test.
            with patch.object(qiskit, "transpile", side_effect=AssertionError("P0 forbidden in T0")):
                value = m.count_native(m.iter_qiskit_native(qc), 3, 2, global_phase=qc.global_phase)
                self.assertEqual(value["metrics"], legacy_metrics(qc))

    def test_qiskit_multiple_register_wire_mapping(self):
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        a, b, c = QuantumRegister(2, "a"), QuantumRegister(1, "b"), ClassicalRegister(2, "c")
        qc = QuantumCircuit(a, b, c); qc.rz(.2, a[1]); qc.cx(a[1], b[0]); qc.measure(b[0], c[1])
        ops = list(m.iter_qiskit_native(qc))
        self.assertEqual(ops[-1], measure(2, 1))
        self.assertEqual(m.count_native(iter(ops), 3, 2)["metrics"], legacy_metrics(qc))

    def test_qiskit_raw_custom_symbolic_and_directive_rejected(self):
        from qiskit import QuantumCircuit
        from qiskit.circuit import Gate, Parameter
        for build in (lambda q:q.crz(.1, 0, 1), lambda q:q.rzz(.1, 0, 1),
                      lambda q:q.cswap(0, 1, 2), lambda q:q.barrier(),
                      lambda q:q.append(Gate("cx", 2, []), [0, 1]),
                      lambda q:q.rz(Parameter("theta"), 0)):
            qc = QuantumCircuit(3);build(qc)
            with self.assertRaises(ValueError):list(m.iter_qiskit_native(qc))

    def test_qiskit_classical_condition_rejected(self):
        from qiskit import QuantumCircuit
        import warnings
        qc = QuantumCircuit(1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            qc.x(0).c_if(qc.clbits[0], 1)
        with self.assertRaises(ValueError):list(m.iter_qiskit_native(qc))


if __name__ == "__main__":
    unittest.main(verbosity=2)
