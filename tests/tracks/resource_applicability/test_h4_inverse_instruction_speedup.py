"""Exact, uncompiled artificial regressions for inverse/instruction reuse."""
import hashlib
import importlib.util
from pathlib import Path
from types import MappingProxyType
import time
import unittest
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate, Gate, Parameter
from qiskit.circuit.library import UnitaryGate

from trottertracks.resource_applicability.h4_geometry import circuits, identity, signal
from trottertracks.resource_applicability.h4_geometry.encoding_memo import ArrayEncodingMemo
from trottertracks.resource_applicability.h4_geometry.instruction_encoding import InstructionEncodingMemo
from trottertracks.resource_applicability.h4_geometry.streaming import LazyList


def reference(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = reference('h4_resume01_speedup_reference')
legacy = reference('h4_run05_serializer_reference')
MEASUREMENTS = {}


def streamed(value, size=65536):
    return b''.join(identity.canonical_chunks(value, size))


def encoded(qc, axis, memo=None):
    memo = InstructionEncodingMemo() if memo is None else memo
    return streamed(circuits.serialize(qc, axis, _encoding_memo=ArrayEncodingMemo(), _instruction_memo=memo))


def fixture(repetitions=1):
    qc = QuantumCircuit(9, 1)
    matrix = np.eye(4, dtype=complex)
    matrix[1, 1] = np.exp(.125j); matrix[2, 2] = np.exp(-.125j)
    base = UnitaryGate(matrix, label='基底🙂')
    definition = QuantumCircuit(3)
    definition.append(base, [0, 1]); definition.cx(2, 0)
    definition.global_phase = -.5
    gate = ControlledGate('人工_ctrl', 3, [], num_ctrl_qubits=1,
                          definition=definition, ctrl_state=0, base_gate=base, label='開制御')
    for i in range(repetitions):
        qc.append(gate, [8, 0, 1])
        qc.rz(-0. if i % 2 else .125, 2)
        qc.cx(8, 3); qc.x(4).c_if(qc.cregs[0], 1)
    qc.global_phase = -.125
    return qc


def artificial_preparation(cached):
    blocks = []
    for i in range(2):
        basis = QuantumCircuit(8)
        for j in range(12):
            basis.rz((i+1)/8, j % 8); basis.cx(j % 7, j % 7+1)
        basis.global_phase = (i+1)/16
        block = {'basis': basis, 'coefficients': {(): .125, (i,): -.5, (i, i+1): .25}}
        if cached:
            block['basis_inverse'] = basis.inverse()
        blocks.append(MappingProxyType(block))
    prep = {'n': 8, 'constant': -.25, 'deterministic': blocks, 'components': []}
    events = [[{'order': 2, 'products': [dict(block=blocks[1], support=(1,), sign=-1)],
                'rotation': dict(block=blocks[0], support=(0, 1), sign=1), 'angle': .125}]]*2
    template = {'q': 2, 'T': .8, 'delta': .4}
    return prep, template, events


class InverseInstructionTests(unittest.TestCase):
    def assert_parent_bytes(self, qc, memo=None):
        for axis in ('cosine', 'sine'):
            self.assertEqual(encoded(qc, axis, memo), streamed(old.serialize(qc, axis, _encoding_memo=ArrayEncodingMemo())))

    def test_paired_full_bytes_digest_phase_controls_labels_condition(self):
        evolution = fixture(8)
        rows = []
        for axis in ('cosine', 'sine'):
            qc = circuits.wrapper(evolution, axis)
            count, depth = dict(qc.count_ops()), qc.depth()
            previous = streamed({'domain': 'h4-numerical-circuit-v1', **old.serialize(qc, axis, _encoding_memo=ArrayEncodingMemo())})
            memo = InstructionEncodingMemo()
            current = streamed({'domain': 'h4-numerical-circuit-v1', **circuits.serialize(qc, axis, _encoding_memo=ArrayEncodingMemo(), _instruction_memo=memo)})
            self.assertEqual(previous, current)
            self.assertEqual(streamed(circuits.serialize(qc, axis, _instruction_memo=InstructionEncodingMemo())), identity.canonical(legacy.serialize(qc, axis)))
            digest = hashlib.sha256(previous).hexdigest()
            self.assertEqual(circuits.numerical_fingerprint(qc, axis, encoding_memo=ArrayEncodingMemo()), digest)
            self.assertEqual((dict(qc.count_ops()), qc.depth()), (count, depth))
            self.assertGreater(memo.hits, 0)
            rows.append(dict(axis=axis, bytes=len(current), sha256=digest, memo_hits=memo.hits, counts=count, depth=depth))
        MEASUREMENTS['paired_bytes'] = rows

    def test_distinct_equal_operations_and_definitions_share_fragments(self):
        qc = QuantumCircuit(2)
        for _ in range(4):
            definition = QuantumCircuit(1); definition.rz(.125, 0)
            op = Gate('same', 1, [], label='定義'); op.definition = definition
            qc.append(op, [0])
        memo = InstructionEncodingMemo()
        self.assert_parent_bytes(qc, memo)
        self.assertGreater(memo.hits, memo.misses)
        self.assertTrue(any(isinstance(value[0], identity.EncodedValue) for value in memo.entries.values()))

    def test_mutated_definition_parameter_and_phase_never_hit_stale_values(self):
        qc = QuantumCircuit(1)
        op = Gate('mutable', 1, []); definition = QuantumCircuit(1); definition.rz(.125, 0)
        op.definition = definition; qc.append(op, [0], copy=False)
        memo = InstructionEncodingMemo(); before = encoded(qc, 'cosine', memo)
        op.definition.rz(.25, 0); op.definition.global_phase = -.0
        after = encoded(qc, 'cosine', memo)
        self.assertNotEqual(before, after); self.assert_parent_bytes(qc, memo)
        op.definition.global_phase = .375
        self.assertNotEqual(after, encoded(qc, 'cosine', memo))
        self.assert_parent_bytes(qc, memo)

    def test_mutated_matrix_value_dtype_signed_zero_and_nonfinite(self):
        memo = InstructionEncodingMemo()
        for value in (0., -0., 1.25):
            qc = QuantumCircuit(1); qc.rz(value, 0)
            self.assert_parent_bytes(qc, memo)
        self.assertNotEqual(encoded(QuantumCircuit(1), 'cosine', memo), encoded(QuantumCircuit(2), 'cosine', memo))
        qc = QuantumCircuit(2); qc.append(UnitaryGate(np.eye(4, dtype=complex)), [0, 1], copy=False)
        matrix = qc.data[0].operation.params[0]
        before = encoded(qc, 'cosine', memo); matrix[0, 0] = -1.
        self.assertNotEqual(before, encoded(qc, 'cosine', memo)); self.assert_parent_bytes(qc, memo)
        matrix[0, 0] = complex(float('nan'), 0)
        with self.assertRaises(identity.Stop): encoded(qc, 'cosine', memo)

    def test_context_axis_order_control_state_label_and_classical_bits(self):
        qc = fixture(1); memo = InstructionEncodingMemo()
        initial = encoded(qc, 'cosine', memo)
        self.assertNotEqual(initial, encoded(qc, 'sine', memo))
        qc.data[0].operation.ctrl_state = 1
        self.assertNotEqual(initial, encoded(qc, 'cosine', memo)); self.assert_parent_bytes(qc, memo)
        qc.data[0].operation.label = '変更🙂'
        self.assert_parent_bytes(qc, memo)
        swapped = QuantumCircuit(9, 1)
        for item in reversed(qc.data):
            swapped.append(item.operation, [qc.find_bit(q).index for q in item.qubits],
                           [qc.find_bit(c).index for c in item.clbits])
        self.assertNotEqual(encoded(qc, 'cosine', memo), encoded(swapped, 'cosine', memo))
        self.assert_parent_bytes(swapped, memo)

    def test_lru_entry_byte_limits_and_disabled_memo(self):
        memo = InstructionEncodingMemo(max_entries=2)
        values = [{'number': x, 'label': '🙂'} for x in (1, 2, 3)]
        a = memo.encode(values[0]); memo.encode(values[1]); self.assertIs(a, memo.encode(values[0]))
        memo.encode(values[2]); missing = memo.misses; memo.encode(values[1])
        self.assertEqual(memo.misses, missing+1)
        self.assertLessEqual(len(memo.entries), 2); self.assertLessEqual(memo.retained_bytes, memo.max_bytes)
        tiny = InstructionEncodingMemo(max_bytes=1)
        self.assertEqual(streamed(tiny.encode(values[0])), identity.canonical(values[0]))
        self.assertEqual(tiny.retained_bytes, 0)
        disabled = InstructionEncodingMemo(max_entries=0)
        self.assertIs(disabled.encode(values[0]), values[0])
        self.assert_parent_bytes(fixture(2), disabled)

    def test_large_array_large_definition_and_label_keep_lazy_streaming(self):
        class NoList(np.ndarray):
            def tolist(self): raise AssertionError('tolist forbidden')
        large = np.zeros((32, 32), dtype=complex).view(NoList)
        memo = InstructionEncodingMemo()
        value = {'parameter': circuits.number(large)}
        self.assertIs(memo.encode(value), value)
        self.assertEqual(streamed(value), streamed({'parameter': old.number(large)}))
        qc = QuantumCircuit(1); definition = QuantumCircuit(1)
        for _ in range(24): definition.rz(.125, 0)
        op = Gate('large', 1, []); op.definition = definition; qc.append(op, [0])
        record = circuits.serialize(qc, 'cosine', _instruction_memo=memo)
        self.assertIsInstance(next(iter(record['instructions']))['definition']['instructions'], LazyList)
        self.assert_parent_bytes(qc, memo)
        long_value = {'label': '🙂'*20000}
        self.assertIs(memo.encode(long_value), long_value)

    def test_recursive_symbolic_and_unsupported_still_stop(self):
        qc = QuantumCircuit(1); qc.rz(Parameter('x'), 0)
        with self.assertRaises(identity.Stop): encoded(qc, 'cosine')
        qc = QuantumCircuit(1); op = Gate('loop', 1, []); op.definition = qc
        qc.append(op, [0], copy=False)
        with self.assertRaises(identity.Stop): encoded(qc, 'cosine')
        class Unknown(Gate): pass
        qc = QuantumCircuit(1); op = Unknown('unknown', 1, []); op.definition = QuantumCircuit(1)
        qc.append(op, [0])
        with self.assertRaises(identity.Stop): encoded(qc, 'cosine')

    def test_default_public_serializer_and_deserializer_compatible(self):
        qc = QuantumCircuit(2, 1); qc.rz(-0., 0); qc.cx(0, 1); qc.measure(1, 0)
        record = circuits.serialize(qc, 'cosine')
        self.assertIsInstance(record['instructions'], LazyList)
        self.assertIsInstance(next(iter(record['instructions'])), dict)
        restored = circuits.deserialize(record)
        self.assertEqual(encoded(qc, 'cosine'), encoded(restored, 'cosine'))

    def test_unicode_chunks_remain_bounded_and_complete(self):
        qc = fixture(1); qc.data[0].operation.label = '日本語🙂"\\'*200
        record = circuits.serialize(qc, 'sine', _instruction_memo=InstructionEncodingMemo())
        chunks = list(identity.canonical_chunks(record, 127))
        self.assertTrue(all(0 < len(chunk) <= 127 for chunk in chunks))
        self.assertEqual(b''.join(chunks), streamed(old.serialize(qc, 'sine')))

    def test_inverse_reuse_has_identical_paired_bytes_counts_and_no_new_inversion(self):
        preparation, template, events = artificial_preparation(True)
        original_inverse = QuantumCircuit.inverse
        calls = []
        def counted(qc, *args, **kwargs):
            calls.append(id(qc)); return original_inverse(qc, *args, **kwargs)
        with patch.object(QuantumCircuit, 'inverse', counted):
            baseline = old.build_evolution(preparation, template, events)
            old_calls = len(calls); calls.clear()
            current = circuits.build_evolution(preparation, template, events)
            self.assertEqual(len(calls), 0); self.assertGreater(old_calls, 0)
        for axis in ('cosine', 'sine'):
            a, b = old.wrapper(baseline, axis), circuits.wrapper(current, axis)
            self.assertEqual(streamed(old.serialize(a, axis)), streamed(old.serialize(b, axis)))
            self.assertEqual(circuits.numerical_fingerprint(a, axis), circuits.numerical_fingerprint(b, axis))
            self.assertEqual((dict(a.count_ops()), a.depth()), (dict(b.count_ops()), b.depth()))
        MEASUREMENTS['inverse_calls_one_artificial_build'] = {'old': old_calls, 'new': 0}

    def test_uncached_blocks_keep_original_fallback_and_basis_unchanged(self):
        prep, template, events = artificial_preparation(False)
        before = [streamed(old.serialize(b['basis'], 'cosine')) for b in prep['deterministic']]
        a, b = old.build_evolution(prep, template, events), circuits.build_evolution(prep, template, events)
        self.assertEqual(streamed(old.serialize(a, 'cosine')), streamed(old.serialize(b, 'cosine')))
        self.assertEqual(before, [streamed(old.serialize(b['basis'], 'cosine')) for b in prep['deterministic']])

    def test_geometry_owned_inverse_precomputed_once_and_builder_does_not_modify_it(self):
        arrays = {'one': np.eye(8), 'G': np.zeros((1, 8, 8)), 'lambdas': np.ones(1),
                  'nuclear': np.array(.5), 'H': np.zeros((1, 1))}
        def block(matrix, lam=None):
            basis = QuantumCircuit(8); basis.rz(.125, 0); basis.cx(0, 1)
            return {'basis': basis, 'coefficients': {(): 0., (0,): .5}}
        with patch.object(circuits, 'diagonalize_block', side_effect=block):
            common = signal.GeometryPreparation(arrays)
        prep = {'n': 8, 'constant': 0., 'deterministic': [common.one, *common.df], 'components': []}
        original = [streamed(old.serialize(b['basis_inverse'], 'cosine')) for b in prep['deterministic']]
        with patch.object(QuantumCircuit, 'inverse', side_effect=AssertionError('inverse must already exist')):
            circuits.build_evolution(prep, {'q': 2, 'T': .8, 'delta': .4}, [[], []])
            circuits.build_evolution(prep, {'q': 1, 'T': .8, 'delta': .8}, [[]])
        self.assertEqual(original, [streamed(old.serialize(b['basis_inverse'], 'cosine')) for b in prep['deterministic']])
        self.assertTrue(all(not a.flags.writeable for a in arrays.values()))

    def test_single_bounded_artificial_timing_no_compile(self):
        # One old/new observation, not a production or benchmark campaign.
        qc = fixture(160)
        start = time.monotonic()
        previous = old.numerical_fingerprint(qc, 'cosine', encoding_memo=ArrayEncodingMemo())
        old_seconds = time.monotonic()-start
        memo = InstructionEncodingMemo(); start = time.monotonic()
        current = circuits.numerical_fingerprint(qc, 'cosine', encoding_memo=ArrayEncodingMemo(), instruction_memo=memo)
        new_seconds = time.monotonic()-start
        self.assertEqual(previous, current)
        MEASUREMENTS['encoding_timing'] = dict(old_seconds=old_seconds, new_seconds=new_seconds,
            speed_ratio=old_seconds/new_seconds, instructions=len(qc.data), memo_hits=memo.hits,
            memo_misses=memo.misses, retained_bytes=memo.retained_bytes, digest=current,
            scope='single artificial uncompiled fixture; production speedup unmeasured')
