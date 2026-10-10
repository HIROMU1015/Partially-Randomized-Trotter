"""Focused artificial IO/launch tests; no production runner or scientific inputs."""
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch

from trottertracks.algorithm_codesign.g10_io import iter_json_bytes, OutputSession, verify_completed
from trottertracks.algorithm_codesign.g10_saved import serial
from trottertracks.algorithm_codesign.g10_launch import verify_launch

ROOT = Path(__file__).resolve().parents[3]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = load_file('g10_artificial_fixtures', ROOT/'scripts/tracks/algorithm_codesign/g10_json_compatibility_fixtures.py')
old_tests = load_file('g10_S2_IO_tests', ROOT/'tests/tracks/algorithm_codesign/test_g10_streaming_io.py')


class KeyCompatibilityTests(unittest.TestCase):
    def check_equivalence(self, value):
        expected = (json.dumps(serial(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
        chunks = list(iter_json_bytes(value))
        actual = b''.join(chunks)
        self.assertEqual(actual, expected)
        self.assertEqual(hashlib.sha256(actual).digest(), hashlib.sha256(expected).digest())
        self.assertTrue(all(len(chunk) <= 32768 for chunk in chunks))
        return json.loads(actual)

    def example(self, name):
        return self.check_equivalence(fixtures.examples()[name])

    def test_provider_integer_labels(self): self.example('provider_integer_labels')
    def test_mixed_keys(self): self.example('mixed_keys')
    def test_collision_integer_first(self):
        actual = self.example('collision_integer_first')
        self.assertEqual(list(actual), ['1', 'middle'])
        self.assertEqual(actual['1'], 'last')
    def test_collision_string_first(self):
        actual = self.example('collision_string_first')
        self.assertEqual(list(actual), ['1', 'middle'])
        self.assertEqual(actual['1'], 'last')
    def test_multiple_collisions(self): self.example('multiple_collisions')
    def test_insertion_order(self):
        self.assertEqual(list(self.example('insertion_order')), ['3', '1', '2'])
    def test_reverse_insertion_order(self):
        self.assertEqual(list(self.example('reverse_order')), ['2', '1', '3'])
    def test_bool_None_keys_use_python_str(self):
        self.assertEqual(list(self.example('bool_None_keys')), ['True', 'False', 'None'])
    def test_float_keys_use_python_str(self): self.example('float_keys')
    def test_tuple_bytes_keys(self): self.example('tuple_bytes_keys')
    def test_nonfinite_keys_stringify_not_value_nan(self): self.example('nonfinite_keys_are_strings')
    def test_nested_event_Fraction_native_ir(self): self.example('nested')
    def test_large_fraction(self): self.example('fractions')
    def test_shared_subtree(self): self.example('shared_subtree')
    def test_Unicode_keys_values(self): self.example('Unicode')
    def test_empty_and_scalars(self): self.example('empty_and_scalars')
    def test_large_typed_payload(self): self.check_equivalence(fixtures.large_typed(2500))
    def test_scalar_numeric_subclasses(self):
        class Int(int):
            def __repr__(self): return 'incorrect custom repr'
        class Float(float):
            def __repr__(self): return 'incorrect custom repr'
        self.check_equivalence([Int(17), Float(-0.0), Float(0.125)])
    def test_pure_custom_key_str(self):
        class Key:
            def __str__(self): return 'shared'
        self.check_equivalence({Key(): 'first', 'middle': None, 'shared': 'last'})
    def test_key_str_exception_propagates(self):
        class Key:
            def __str__(self): raise ValueError('artificial key conversion failure')
        with self.assertRaisesRegex(ValueError, 'artificial key conversion failure'):
            list(iter_json_bytes({Key(): 'value'}))
    def test_graph_and_key_insertion_order_not_mutated(self):
        value = fixtures.examples()['shared_subtree']
        before = copy.deepcopy(value)
        shared = value['a']
        keys = list(shared['provider_calls'])
        self.check_equivalence(value)
        self.assertEqual(value, before)
        self.assertIs(value['a'], value['b'][0])
        self.assertIs(value['b'][0], value['b'][1])
        self.assertEqual(list(shared['provider_calls']), keys)
    def test_overwritten_values_still_validated(self):
        for invalid in (float('nan'), float('inf'), -float('inf'), object(), b'value'):
            with self.subTest(type=type(invalid).__name__), self.assertRaises((TypeError, ValueError)):
                list(iter_json_bytes({1: invalid, '1': 'finite replacement'}))
    def test_collision_does_not_hide_cycle(self):
        cyc = []; cyc.append(cyc)
        with self.assertRaisesRegex(ValueError, 'circular'):
            list(iter_json_bytes({1: cyc, '1': 'replacement'}))
    def test_dict_cycle(self):
        value = {1: None}; value[1] = value
        with self.assertRaises(ValueError): list(iter_json_bytes(value))
    def test_normalized_map_guard_is_periodic(self):
        calls = []
        def check():
            calls.append(1)
            if len(calls) == 7: raise TimeoutError('artificial key-map timeout')
        with self.assertRaises(TimeoutError): list(iter_json_bytes({i: 0 for i in range(3000)}, check))
    def test_deterministic_nested_case_matrix(self):
        # Finite artificial encoder coverage, not scientific sampling/grid.
        for n in range(128):
            value = {n: F(n-7, 11), str(n): (n, {'calls': {n % 3: 2}}),
                     'Unicode': '漢😀'*((n % 3)+1)}
            self.check_equivalence([value, (value, None, -0.0)])
    def test_normal_output_session_with_key_collisions(self):
        with tempfile.TemporaryDirectory(prefix='g10-artificial-IO-') as directory:
            root = Path(directory)
            marker = root/'one_shot_consumed.json'; marker.write_bytes(b'artificial-marker')
            guard = old_tests.Guard()
            io = OutputSession(root, 1000000, guard, {'scope': 'synthetic only'})
            value = fixtures.examples()['nested']
            identity = io.write_result(value); io.success(identity)
            self.assertEqual(verify_completed(root), identity)
            self.assertEqual(marker.read_bytes(), b'artificial-marker')
            self.assertEqual(io.final.read_bytes(),
                (json.dumps(serial(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode())
    def test_key_conversion_failure_receipted(self):
        class Key:
            def __str__(self): raise ValueError('artificial-key')
        with tempfile.TemporaryDirectory(prefix='g10-artificial-IO-') as directory:
            root = Path(directory); marker = root/'one_shot_consumed.json'
            marker.write_bytes(b'artificial-marker')
            io = OutputSession(root, 1000000, old_tests.Guard(), {'scope': 'synthetic only'})
            try:
                io.write_result({Key(): 'value'})
            except ValueError as exc:
                io.failure(exc, 0, 0, 0)
            else:
                self.fail('key error not raised')
            self.assertFalse((root/'COMPLETED.v2').exists())
            self.assertEqual(marker.read_bytes(), b'artificial-marker')
            self.assertFalse(json.loads((root/'failure_receipt_v2.json').read_text())['scientific_result_committed'])


class PreparationLaunchTests(unittest.TestCase):
    def test_pending_v3_refused_before_science_or_git(self):
        contract = ROOT/'artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3/contract_v3.json'
        with patch('trottertracks.algorithm_codesign.g10_launch.git', side_effect=AssertionError('Git must not be reached')):
            with self.assertRaisesRegex(PermissionError, 'pending'):
                verify_launch(ROOT, contract, '0'*40)
    def test_no_scientific_imports(self):
        import sys
        self.assertFalse(any(name.startswith(('numpy', 'mpmath', 'pygridsynth')) for name in sys.modules))


def load_tests(loader, standard, pattern):
    # Keep all 46 applicable S2 IO/guard/launch/lifetime tests unchanged.
    # Only the superseded blanket nonstring-key rejection test is excluded.
    def flatten(suite):
        for test in suite:
            if isinstance(test, unittest.TestSuite): yield from flatten(test)
            else: yield test
    inherited = [test for test in flatten(loader.loadTestsFromModule(old_tests))
                 if test._testMethodName != 'test_nonstring_key_rejected']
    return unittest.TestSuite([*inherited, standard])


if __name__ == '__main__': unittest.main()
