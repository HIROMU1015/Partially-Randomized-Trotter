"""Synthetic I/O tests only: no runner, matrix, synthesis, sampling or LP."""
import ast
import copy
import hashlib
import json
import signal
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch

from trottertracks.algorithm_codesign.g10_io import (
    IOBudgetGuard, OutputSession, RECEIPT_CAP, TERMINAL_RESERVE,
    iter_json_bytes, validate_tree, verify_completed,
    protected_check_streaming,
)
from trottertracks.algorithm_codesign.g10_launch import validate_binding

ROOT = Path(__file__).resolve().parents[3]


def legacy(value):
    # Independent copy of the unchanged, pure old serialization contract.
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(k): legacy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [legacy(v) for v in value]
    return value


class Guard:
    def __init__(self):
        self.calls = 0
        self.trigger = None
        self.suspended = False

    def check(self):
        self.calls += 1
        if self.trigger and self.trigger():
            raise MemoryError('synthetic RSS guard; no retry')

    def usage(self):
        return {'wall_seconds': 0.0, 'cpu_seconds': 0.0, 'peak_RSS_KiB': 1, 'processes': 1}

    def suspend_for_failure_receipt(self):
        self.suspended = True


class EncoderTests(unittest.TestCase):
    def assert_equal(self, value):
        expected = (json.dumps(legacy(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
        chunks = list(iter_json_bytes(value))
        self.assertEqual(b''.join(chunks), expected)
        self.assertTrue(all(len(x) <= 32768 for x in chunks))

    def test_fraction(self): self.assert_equal({'f': F(-11, 13), 'zero': F(0)})
    def test_very_large_fraction(self): self.assert_equal({'f': F(2**1600+1, 2**1200)})
    def test_tuple(self): self.assert_equal({'native_ir': (('R', F(1, 3)), ('CX', 0, 1))})
    def test_lists(self): self.assert_equal([None, True, False, 1, -7, 0.1, -0.0, [], {}])
    def test_unicode(self): self.assert_equal({'研究😀': 'α\n"\\漢字'})
    def test_large_unicode(self): self.assert_equal({'s': '😀漢\n"'*25000})
    def test_shared_subtree(self):
        child = {'x': (F(7, 9),)}
        self.assert_equal([child, child])
    def test_nan(self):
        with self.assertRaises(ValueError): list(iter_json_bytes({'x': float('nan')}))
    def test_infinity(self):
        with self.assertRaises(ValueError): list(iter_json_bytes({'x': float('inf')}))
    def test_negative_infinity(self):
        with self.assertRaises(ValueError): list(iter_json_bytes({'x': -float('inf')}))
    def test_unsupported(self):
        with self.assertRaises(TypeError): list(iter_json_bytes({'x': object()}))
    def test_bytes_rejected(self):
        with self.assertRaises(TypeError): list(iter_json_bytes({'x': b'x'}))
    def test_nonstring_key_rejected(self):
        for key in (True, 1, None, F(1, 3)):
            with self.subTest(key=key), self.assertRaises(TypeError):
                list(iter_json_bytes({key: 'x'}))
    def test_cycle(self):
        value = []; value.append(value)
        with self.assertRaises(ValueError): list(iter_json_bytes(value))
    def test_nested_guard(self):
        def fail(): raise TimeoutError('synthetic')
        with self.assertRaises(TimeoutError): validate_tree({'a': list(range(3000))}, fail)
    def test_no_mutation(self):
        value = {'events': [{'native_ir': [('R', F(1, 7))]}]}
        original = copy.deepcopy(value)
        list(iter_json_bytes(value))
        self.assertEqual(value, original)


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        (self.directory/'one_shot_consumed.json').write_bytes(b'unchanged-marker')
        self.guard = Guard()
        self.io = OutputSession(self.directory, 1000000, self.guard, {'source_commit': 's'*40})

    def assert_failure(self, exc):
        self.io.failure(exc, 17, 27, 19)
        receipt = json.loads((self.directory/'failure_receipt_v2.json').read_text())
        self.assertEqual(receipt['status'], 'G10_TECHNICAL_INCONCLUSIVE')
        self.assertFalse(receipt['scientific_result_committed'])
        self.assertFalse(receipt['prefix_rows_usable_for_final_research_decision'])
        self.assertTrue(receipt['mandatory_STOP'])
        self.assertEqual(receipt['retries'], 0)
        self.assertLessEqual((self.directory/'failure_receipt_v2.json').stat().st_size, RECEIPT_CAP)
        self.assertEqual((self.directory/'one_shot_consumed.json').read_bytes(), b'unchanged-marker')
        with self.assertRaises(PermissionError): verify_completed(self.directory)
        return receipt

    def test_success(self):
        value = {'rows': [{'native_ir': [('R', F(1, 9))]}], '研究': True}
        identity = self.io.write_result(value)
        self.assertEqual(identity['sha256'], hashlib.sha256(b''.join(iter_json_bytes(value))).hexdigest())
        self.assertFalse(self.io.partial.exists())
        with self.assertRaises(PermissionError): verify_completed(self.directory)
        self.io.success(identity)
        self.assertEqual(verify_completed(self.directory), identity)
        self.assertGreater(self.guard.calls, 10)
    def test_missing_token_rejected(self):
        identity = self.io.write_result({'x': 1})
        self.io._small('STOP.json', {'scientific_result_committed': True}, True)
        with self.assertRaises(PermissionError): verify_completed(self.directory)
    def test_corruption_rejected(self):
        identity = self.io.write_result({'x': 1}); self.io.success(identity)
        self.io.final.write_bytes(b'wrong')
        with self.assertRaises(PermissionError): verify_completed(self.directory)
    def test_output_cap(self):
        self.io.cap = TERMINAL_RESERVE+600
        with self.assertRaises(RuntimeError) as context: self.io.write_result({'s': 'x'*1000})
        self.assertLessEqual(self.io.bytes_used, self.io.cap)
        self.assert_failure(context.exception)
    def test_exact_room_boundary(self):
        self.io.cap = self.io.bytes_used+TERMINAL_RESERVE+10
        self.io._room(10)
        with self.assertRaises(RuntimeError): self.io._room(11)
    def test_guard_during_write(self):
        self.guard.trigger = lambda: self.io.payload_bytes > 0
        with self.assertRaises(MemoryError) as context: self.io.write_result({'s': 'x'*90000})
        receipt = self.assert_failure(context.exception)
        self.assertEqual(receipt['partial_output']['written_bytes'], self.io.partial.stat().st_size)
        self.assertEqual(receipt['partial_output']['written_prefix_sha256'],
                         hashlib.sha256(self.io.partial.read_bytes()).hexdigest())
        self.assertFalse(self.io.final.exists())
    def test_guard_before_flush(self):
        self.guard.trigger = lambda: self.io.phase == 'flush_fsync'
        with self.assertRaises(MemoryError) as context: self.io.write_result({'x': 1})
        self.assert_failure(context.exception)
    def test_guard_after_close(self):
        self.guard.trigger = lambda: self.io.phase == 'close'
        with self.assertRaises(MemoryError) as context: self.io.write_result({'x': 1})
        self.assert_failure(context.exception)
    def test_guard_after_promotion(self):
        self.guard.trigger = lambda: self.io.promoted
        with self.assertRaises(MemoryError) as context: self.io.write_result({'x': 1})
        self.assertTrue(self.io.final.exists())
        self.assert_failure(context.exception)
    def test_guard_after_stop(self):
        identity = self.io.write_result({'x': 1})
        self.guard.trigger = lambda: (self.directory/'STOP.json').exists()
        with self.assertRaises(MemoryError) as context: self.io.success(identity)
        self.assert_failure(context.exception)
        self.assertFalse((self.directory/'COMPLETED.v2').exists())
    def test_guard_after_token_close(self):
        identity = self.io.write_result({'x': 1})
        self.guard.trigger = lambda: (self.directory/'COMPLETED.v2').exists()
        with self.assertRaises(MemoryError) as context: self.io.success(identity)
        self.assert_failure(context.exception)
        self.assertFalse((self.directory/'COMPLETED.v2').exists())
    def test_failed_receipt_still_invalidates_own_token(self):
        identity = self.io.write_result({'x': 1})
        self.io.success(identity)
        with patch.object(self.io, '_small', side_effect=OSError('receipt disk failure')):
            with self.assertRaises(OSError): self.io.failure(MemoryError('synthetic'), 1, 0, 0)
        self.assertFalse((self.directory/'COMPLETED.v2').exists())
        with self.assertRaises(PermissionError): verify_completed(self.directory)
    def test_foreign_token_not_removed(self):
        (self.directory/'COMPLETED.v2').write_bytes(b'old-token')
        self.io.failure(RuntimeError('synthetic'), 1, 0, 0)
        self.assertEqual((self.directory/'COMPLETED.v2').read_bytes(), b'old-token')
    def test_fsync_failure(self):
        with patch('trottertracks.algorithm_codesign.g10_io.os.fsync', side_effect=OSError('disk')):
            with self.assertRaises(OSError) as context: self.io.write_result({'x': 1})
        self.assert_failure(context.exception)
    def inject_stream_fault(self, fault):
        original = Path.open
        io = self.io
        class FaultStream:
            def __init__(self, wrapped): self.wrapped = wrapped
            def __enter__(self): return self
            def __exit__(self, *args):
                self.wrapped.close()
                if fault == 'close': raise OSError('synthetic close failure')
            def write(self, raw):
                if fault == 'write': raise OSError('synthetic write failure')
                if fault == 'short': return self.wrapped.write(raw[:7])
                return self.wrapped.write(raw)
            def flush(self):
                if fault == 'flush': raise OSError('synthetic flush failure')
                return self.wrapped.flush()
            def fileno(self): return self.wrapped.fileno()
        def patched(path, *args, **kwargs):
            stream = original(path, *args, **kwargs)
            return FaultStream(stream) if path == io.partial and args[0] == 'xb' else stream
        with patch.object(Path, 'open', patched):
            with self.assertRaises(OSError) as context: self.io.write_result({'x': 'artificial'})
        self.assert_failure(context.exception)
        self.assertFalse(self.io.final.exists())
    def test_write_failure(self): self.inject_stream_fault('write')
    def test_short_write(self): self.inject_stream_fault('short')
    def test_flush_failure(self): self.inject_stream_fault('flush')
    def test_close_failure(self): self.inject_stream_fault('close')
    def test_promotion_collision(self):
        self.io.final.write_bytes(b'old-result')
        with self.assertRaises(FileExistsError) as context: self.io.write_result({'x': 1})
        self.assertEqual(self.io.final.read_bytes(), b'old-result')
        self.assert_failure(context.exception)
    def test_partial_collision(self):
        self.io.partial.write_bytes(b'old-partial')
        with self.assertRaises(FileExistsError) as context: self.io.write_result({'x': 1})
        self.assertEqual(self.io.partial.read_bytes(), b'old-partial')
        self.assert_failure(context.exception)
    def test_unsupported_receipted_without_fallback(self):
        with self.assertRaises(TypeError) as context: self.io.write_result({'x': object()})
        self.assert_failure(context.exception)
    def test_bounded_reason(self): self.assert_failure(RuntimeError('x'*100000))
    def test_existing_failure_not_overwritten(self):
        self.assert_failure(RuntimeError('first'))
        before = (self.directory/'failure_receipt_v2.json').read_bytes()
        with self.assertRaises(FileExistsError): self.io.failure(RuntimeError('second'), 0, 0, 0)
        self.assertEqual((self.directory/'failure_receipt_v2.json').read_bytes(), before)
    def test_traceback_recorded_and_detached(self):
        try: raise ValueError('synthetic')
        except ValueError as exc:
            receipt = self.assert_failure(exc)
            self.assertIsNone(exc.__traceback__)
            self.assertEqual(receipt['exception_location'][-1]['function'], 'test_traceback_recorded_and_detached')
    def test_emergency_retains_hard_limits(self):
        guard = IOBudgetGuard.__new__(IOBudgetGuard)
        with patch('trottertracks.algorithm_codesign.g10_io.signal.setitimer') as timer:
            guard.suspend_for_failure_receipt()
        timer.assert_called_once_with(signal.ITIMER_REAL, 0)


class LifetimeAndLaunchTests(unittest.TestCase):
    def test_streaming_protected_check_equivalence(self):
        from trottertracks.algorithm_codesign.g7_launch import protected_check
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'full').write_bytes(b'x'*90000)
            (root/'prefix').write_bytes(b'old-prefix appended')
            ledger = {name: {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
                      for name, value in (('full', b'x'*90000), ('prefix', b'old-prefix'))}
            (root/'ledger.json').write_text(json.dumps(ledger))
            c = {'protected_ledger': 'ledger.json', 'append_only_paths': ['prefix']}
            self.assertEqual(protected_check_streaming(root, c), protected_check(root, c))
            (root/'full').write_bytes(b'changed')
            self.assertEqual(protected_check_streaming(root, c), protected_check(root, c))
    def test_npz_rejected_before_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'ledger.json').write_text(json.dumps({'not-present.npz': {'bytes': 0, 'sha256': 'x'}}))
            with self.assertRaises(PermissionError):
                protected_check_streaming(root, {'protected_ledger': 'ledger.json', 'append_only_paths': []})
    def test_event_alias_survives_pending_clear(self):
        events = [{'native_ir': [('CX', 0, 1)], 'budget': {'m': 7}}]
        pending = [(7, 'synthetic', events)]
        result = {'rows': [{'events': pending[0][2]}]}
        pending.clear(); del pending, events
        self.assertEqual(result['rows'][0]['events'][0]['native_ir'], [('CX', 0, 1)])
    def test_anchor_copy_preserves_all_fields(self):
        old = {'events': [{'native_ir': [('R', F(1, 7))]}], 'budget': {'N': 2},
               'synthesis_identity': 'fixed', 'original_G9_budget': {'N': 1}}
        anchor = copy.deepcopy(old)
        anchor['budget']['N'] = 3
        self.assertEqual(old['budget']['N'], 2)
        del old
        self.assertEqual(anchor['events'][0]['native_ir'][0][1], F(1, 7))
        self.assertEqual(anchor['original_G9_budget'], {'N': 1})
    def test_pending_binding_refused(self):
        with self.assertRaises(PermissionError):
            validate_binding({'status': 'PENDING', 'science_execution_authorized': False},
                             'hash', 's', 'h', ['s'], ['auth'], False, {'auth'})
    def test_direct_child_binding(self):
        source = 'a'*40
        auth = {'source_commit': source, 'status': 'APPROVED_FOR_ONE_G10_RUN',
                'science_execution_authorized': True, 'runs': 1, 'retries': 0,
                'mandatory_STOP': True, 'explicit_execution_instruction': 'synthetic test only',
                'contract_sha256': 'hash'}
        validate_binding(auth, 'hash', source, 'b'*40, [source], ['new-auth'], False, {'new-auth'})
        for parents, changed, dirty in (([source, 'c'*40], ['new-auth'], False),
                                         ([source], ['old-marker'], False),
                                         ([source], ['new-auth'], True)):
            with self.subTest(parents=parents, changed=changed, dirty=dirty), self.assertRaises(PermissionError):
                validate_binding(auth, 'hash', source, 'b'*40, parents, changed, dirty, {'new-auth'})


if __name__ == '__main__': unittest.main()
