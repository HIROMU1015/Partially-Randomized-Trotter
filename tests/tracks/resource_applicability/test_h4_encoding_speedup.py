"""Small, uncompiled artificial fixtures for exact encoding reuse."""
import dataclasses
import hashlib
import importlib.util
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate, Parameter
from qiskit.circuit.library import UnitaryGate
from trottertracks.resource_applicability.h4_geometry import circuits, execution, identity
from trottertracks.resource_applicability.h4_geometry.encoding_memo import ArrayEncodingMemo

spec = importlib.util.spec_from_file_location('encoding_oracle', Path(__file__).with_name('h4_run05_serializer_reference.py'))
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
MEASUREMENTS = {}


def streamed(value, chunk_size=65536):
    return b''.join(identity.canonical_chunks(value, chunk_size))


def fixture(repetitions=1):
    """8 system qubits + ancilla; four synthetic 4x4 matrix values only."""
    qc = QuantumCircuit(9)
    for i in range(repetitions):
        matrix = np.eye(4, dtype=np.complex128)
        matrix[1, 1] = np.exp(1j * (i % 4 + 1)/8)
        matrix[2, 2] = matrix[1, 1].conjugate()
        gate = UnitaryGate(matrix, label='人工Gaussian🙂')
        qc.append(gate, [i % 7, i % 7 + 1])
        qc.rz(-0., i % 8)
        qc.cx(8, i % 8)
    qc.global_phase = -.125
    return qc


class EncodingTests(unittest.TestCase):
    def test_exact_array_bytes_dtype_shape_signed_zero_and_order(self):
        memo = ArrayEncodingMemo()
        arrays = [np.array(-0.), np.array(complex(-0., -0.)), np.zeros((0, 2)),
                  np.array([True, False]), np.array([[1, -2]], dtype='>i4'),
                  np.array([0., -0., 1.25], dtype='>f4'),
                  np.arange(12, dtype=np.uint64).reshape(3, 4)[:, ::-1],
                  np.asfortranarray(np.array([[1+2j, 3-4j], [5+6j, complex(-0., -0.)]]))]
        for array in arrays:
            with self.subTest(dtype=array.dtype, shape=array.shape):
                encoded = circuits.number(array, _encoding_memo=memo)
                self.assertEqual(streamed(encoded), identity.canonical(old.number(array)))
                self.assertEqual(identity.canonical(encoded), streamed(encoded))

    def test_equal_values_hit_across_distinct_arrays(self):
        memo = ArrayEncodingMemo()
        array = np.eye(4, dtype=np.complex128)
        first = memo.encode(array)
        self.assertIs(first, memo.encode(array.copy(order='F')))
        self.assertEqual((memo.hits, memo.misses), (1, 1))

    def test_mutation_never_uses_stale_encoding(self):
        memo = ArrayEncodingMemo()
        array = np.array([0., 1.])
        first = streamed(memo.encode(array))
        array[0] = -0.
        second = streamed(memo.encode(array))
        self.assertNotEqual(first, second)
        array[1] = 2.
        self.assertEqual(streamed(memo.encode(array)), identity.canonical(old.number(array)))

    def test_nonfinite_after_cache_hit_stops_without_insertion(self):
        memo = ArrayEncodingMemo()
        array = np.array([1+0j])
        memo.encode(array); memo.encode(array)
        for bad in (complex(float('nan'), 0), complex(1, float('inf'))):
            array[0] = bad
            with self.assertRaises(identity.Stop): memo.encode(array)
        self.assertEqual(len(memo.entries), 1)

    def test_lru_entry_and_byte_limits(self):
        memo = ArrayEncodingMemo(max_entries=2, max_bytes=3000)
        a, b, c = (np.array([n]) for n in range(3))
        memo.encode(a); memo.encode(b); memo.encode(a); memo.encode(c)
        self.assertLessEqual(len(memo.entries), 2)
        self.assertLessEqual(memo.retained_bytes, 3000)
        misses = memo.misses
        memo.encode(b)
        self.assertEqual(memo.misses, misses+1)
        tiny = ArrayEncodingMemo(max_bytes=1)
        self.assertEqual(streamed(tiny.encode(a)), identity.canonical(old.number(a)))
        self.assertEqual(tiny.retained_bytes, 0)
        self.assertIsNone(ArrayEncodingMemo(max_entries=0).encode(a))

    def test_large_arrays_still_stream_without_tolist(self):
        class NoList(np.ndarray):
            def tolist(self): raise AssertionError('tolist forbidden')
        array = np.zeros((32, 32), dtype=complex).view(NoList)
        memo = ArrayEncodingMemo()
        encoded = circuits.number(array, _encoding_memo=memo)
        chunks = list(identity.canonical_chunks(encoded, 1024))
        self.assertTrue(all(0 < len(x) <= 1024 for x in chunks))
        self.assertEqual(b''.join(chunks), streamed(circuits.number(array)))
        self.assertEqual(len(memo.entries), 0)

    def test_fragment_validation_immutability_and_unicode_chunking(self):
        value = {'text': '日本語🙂"\\'*100, 'number': complex(-0., -0.)}
        encoded = identity.EncodedValue(value)
        self.assertEqual(streamed(encoded, 7), identity.canonical(value))
        with self.assertRaises(dataclasses.FrozenInstanceError): encoded.text = 'null'
        with self.assertRaises(identity.Stop): identity.EncodedValue(float('nan'))
        self.assertEqual(streamed(identity.EncodedValue('null')), b'"null"')

    def test_symbolic_and_unsupported_values_stop(self):
        memo = ArrayEncodingMemo()
        for value in (Parameter('p'), np.array(['invalid'])):
            with self.assertRaises(identity.Stop):
                streamed(circuits.number(value, _encoding_memo=memo))
        qc = QuantumCircuit(1); qc.rz(Parameter('p'), 0)
        with self.assertRaises(identity.Stop):
            circuits.numerical_fingerprint(qc, 'cosine', encoding_memo=memo)

    def test_paired_circuit_old_new_full_bytes_digest_and_gate_counts(self):
        memo = ArrayEncodingMemo()
        evolution = fixture(16)
        rows = []
        for axis in ('cosine', 'sine'):
            qc = circuits.wrapper(evolution, axis)
            counts, depth = dict(qc.count_ops()), qc.depth()
            before = identity.canonical({'domain': 'h4-numerical-circuit-v1', **old.serialize(qc, axis)})
            after = streamed({'domain': 'h4-numerical-circuit-v1', **circuits.serialize(qc, axis, _encoding_memo=memo)})
            self.assertEqual(before, after)
            digest = hashlib.sha256(before).hexdigest()
            self.assertEqual(circuits.numerical_fingerprint(qc, axis, encoding_memo=memo), digest)
            self.assertEqual((dict(qc.count_ops()), qc.depth()), (counts, depth))
            rows.append(dict(axis=axis, bytes=len(after), sha256=digest, counts=counts, depth=depth))
        self.assertGreater(memo.hits, 0)
        MEASUREMENTS['paired_circuit'] = rows

    def test_nested_open_control_phase_labels_and_condition(self):
        base = UnitaryGate(np.eye(4, dtype=complex), label='基底🙂')
        definition = QuantumCircuit(3)
        definition.append(base, [0, 1]); definition.cx(2, 0)
        definition.global_phase = -.5
        ctrl = ControlledGate('synthetic_ctrl', 3, [], num_ctrl_qubits=1,
                              definition=definition, ctrl_state=0, base_gate=base, label='開制御')
        qc = QuantumCircuit(9, 1)
        qc.append(ctrl, [8, 0, 1]); qc.x(2).c_if(qc.cregs[0], 1)
        qc.global_phase = .375
        memo = ArrayEncodingMemo()
        for axis in ('cosine', 'sine'):
            self.assertEqual(streamed(circuits.serialize(qc, axis, _encoding_memo=memo)),
                             identity.canonical(old.serialize(qc, axis)))

    def test_candidate_generator_shares_memo_and_preserves_order(self):
        evolution = fixture(4)
        captured = []
        real_fingerprint = circuits.numerical_fingerprint
        def fingerprint(qc, axis, *, encoding_memo):
            captured.append(encoding_memo)
            return real_fingerprint(qc, axis, encoding_memo=encoding_memo)
        class Run:
            def pulse(self): pass
            def phase(self, name): pass
        with patch.object(circuits, 'build_evolution', return_value=evolution), \
             patch.object(circuits, 'numerical_fingerprint', side_effect=fingerprint), \
             patch('trottertracks.resource_applicability.h4_geometry.signal.sample_events', return_value=([[]], None)), \
             patch.object(execution, 'wire', side_effect=lambda ident, axis, seed, index, digest: (axis, seed, index, digest)):
            records = list(execution.candidate_wrapper_jobs(Run(), {}, [11, 22], {'components': []}, {'q': 1}))
        self.assertEqual([r[0][:3] for r in records], [('cosine', 11, 0), ('sine', 11, 0), ('cosine', 22, 1), ('sine', 22, 1)])
        self.assertTrue(all(m is captured[0] for m in captured))
        self.assertGreater(captured[0].hits, captured[0].misses)

    def test_bounded_artificial_encoding_timing(self):
        # One measurement per path; no transpile or repeat benchmark campaign.
        qc = circuits.wrapper(fixture(600), 'cosine')
        start = time.monotonic()
        baseline = circuits.numerical_fingerprint(qc, 'cosine')
        old_seconds = time.monotonic()-start
        memo = ArrayEncodingMemo()
        start = time.monotonic()
        new = circuits.numerical_fingerprint(qc, 'cosine', encoding_memo=memo)
        new_seconds = time.monotonic()-start
        self.assertEqual(baseline, new)
        MEASUREMENTS['encoding_timing'] = dict(old_seconds=old_seconds, new_seconds=new_seconds,
            ratio=old_seconds/new_seconds, instructions=len(qc.data), digest=baseline,
            cache_hits=memo.hits, cache_misses=memo.misses, retained_bytes=memo.retained_bytes,
            scope='single uncompiled artificial fixture; not production speedup')
