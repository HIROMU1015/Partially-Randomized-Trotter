"""Artificial-only equivalence and process STOP tests; no science imports."""
from dataclasses import replace
from fractions import Fraction as F
import importlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
P = importlib.import_module('trottertracks.algorithm_codesign.g6_w2c_b3_performance_pilot')
import test_g6_w2c_b3_f01_provenance as F01


class Equivalence(unittest.TestCase):
    def test_all_ascii_canonical_bytes_including_del(self):
        for i in range(128):
            payload = {chr(i): chr(i)}
            self.assertEqual(P.canonical_fast(payload, 'orjson'), P.D.audit.canonical(P.D.exact(payload)))

    def test_buffer_crash_changes_accepted_prefix_guarantee(self):
        with tempfile.TemporaryDirectory(prefix='ARTIFICIAL_B3_PERF_') as tmp:
            direct = P.buffer_crash_probe(Path(tmp)/'direct.jsonl', 0)
            buffered = P.buffer_crash_probe(Path(tmp)/'buffered.jsonl', 65536)
            self.assertEqual(direct['accepted_rows'], 2)
            self.assertEqual(direct['retained']['rows'], 2)
            self.assertEqual(buffered['accepted_rows'], 2)
            self.assertEqual(buffered['retained']['rows'], 0)

    def test_canonical_unicode_integer_keys_collisions_bigints_and_surrogates(self):
        rng = random.Random(19)
        inputs = [None, True, False, 0, -1, F(3, 7), {'日本語': '角度\n\t\"\\'},
                  {'surrogate': '\ud800'}, {1: 'overwritten', '1': 'last'},
                  {'huge': F((1 << 8192)+1, (1 << 4096)+3)}, {False: [1, 2, (F(1, 3),)]}]
        for _ in range(100):
            inputs.append({'z': [rng.getrandbits(256), F(rng.getrandbits(128), rng.getrandbits(128)+1)],
                           'a': {'é': '🙂', 'b': rng.choice([True, False, None])}})
        for value in inputs:
            expected = P.D.audit.canonical(P.D.exact(value))
            for backend in ('python', 'orjson'):
                self.assertEqual(P.canonical_fast(value, backend), expected)

    def test_unsupported_floats_rejected(self):
        for backend in ('python', 'orjson'):
            with self.assertRaises(P.D.audit.InvalidEvidence):
                P.canonical_fast({'bad': 0.25}, backend)

    def test_exact_endpoint_bounds_partial_and_remainder(self):
        rng = random.Random(41)
        values = [F(0), F(1, 2), F(2, 3), F(3, 4)]
        values += [F(rng.getrandbits(b), (1 << (b+1))+7) for b in (8, 32, 64, 128, 256)]
        for z in values:
            for terms in (1, 2, 4, 12, 32):
                expected = P.atan_reference(z, terms)
                self.assertEqual(P.atan_recurrence(z, terms), expected)
                self.assertEqual(P.atan_gmp(z, terms), expected)

    def test_science_import_is_blocked(self):
        before = len(P.BLOCKED_IMPORTS)
        try:
            with self.assertRaisesRegex(RuntimeError, 'FORBIDS_SCIENCE_IMPORT'):
                importlib.import_module('trottertracks.algorithm_codesign.g7_generator')
        finally:
            del P.BLOCKED_IMPORTS[before:]
        P.assert_no_science()

    def test_serial_parallel_native_and_buffered_content_trace_bits(self):
        with tempfile.TemporaryDirectory(prefix='ARTIFICIAL_B3_PERF_') as tmp:
            base = P.scheduler(P.tasks_for(n=20), 1, Path(tmp)/'serial')
            for variant, workers, buf in [('reference', 4, 0), ('recurrence', 4, 0),
                                          ('gmp', 4, 0), ('orjson', 4, 0), ('orjson', 4, 65536)]:
                tasks = [replace(t, variant=variant, buffer_bytes=buf) for t in P.tasks_for(n=20)]
                result = P.scheduler(tasks, workers, Path(tmp)/(variant+str(buf)))
                for a, b in zip(base['workers_results'], result['workers_results']):
                    self.assertEqual(b['status'], 'OK')
                    self.assertEqual(a['result'], b['result'])
                    self.assertEqual(a['normalized_file_sha256'], b['normalized_file_sha256'])
                    self.assertEqual(a['normalized_chain_sha256'], b['normalized_chain_sha256'])
                    self.assertEqual(b['readback']['rows'], b['completed'])

    def test_layer_and_pass_event_trace_and_bit_identity(self):
        with tempfile.TemporaryDirectory(prefix='ARTIFICIAL_B3_PERF_') as tmp:
            results = []
            for layer, instrumented in [('A', False), ('B', False), ('B', True)]:
                t = P.Task('ARTIFICIAL_test', 'full_return', layer, 16, instrumented=instrumented)
                results.append(P.perform_task(t, tmp))
                (Path(tmp)/(t.key+'.jsonl')).unlink()
            for r in results[1:]:
                self.assertEqual(results[0]['result']['trace_sha256'], r['result']['trace_sha256'])
                self.assertEqual(results[0]['result']['axes'], r['result']['axes'])

    def test_no_retry_resume_or_output_collision(self):
        with tempfile.TemporaryDirectory(prefix='ARTIFICIAL_B3_PERF_') as tmp:
            path = Path(tmp)/'spent'
            tasks = P.tasks_for(1, n=2)
            P.scheduler(tasks, 1, path)
            with self.assertRaises(FileExistsError):
                P.scheduler(tasks, 1, path)
            with self.assertRaises(ValueError):
                P.scheduler(tasks+tasks, 2, Path(tmp)/'duplicate')

    def test_failure_stop_and_f01_prefix(self):
        with tempfile.TemporaryDirectory(prefix='ARTIFICIAL_B3_PERF_') as tmp:
            tasks = [replace(t, track_starts=True) for t in P.tasks_for(16, n=24)]
            tasks[0] = replace(tasks[0], failure_trial=3)
            r = P.scheduler(tasks, 8, Path(tmp)/'fail')
            self.assertGreater(r['failure_ns'], 0)
            self.assertEqual(r['starts_after_failure'], 0)
            self.assertGreaterEqual(len(r['not_launched']), 8)
            failure = next(x for x in r['workers_results'] if x['status'] == 'FAILED')['failure']
            self.assertEqual(sum(failure['completed_prefix'].values()), 3)
            self.assertIsNone(failure['inflight_event'])
            self.assertFalse(failure['current_attempt']['result_acquired'])
            self.assertFalse(failure['resume'])
            self.assertEqual(failure['retry'], 0)


class PythonStreamF01(F01.F01):
    backend = 'python'

    def setUp(self):
        backend = self.backend
        class Selected(P.PrototypeStream):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, backend=backend, **kwargs)
        self.stream_patch = patch.object(P.D, 'Stream', Selected)
        self.stream_patch.start()
        self.addCleanup(self.stream_patch.stop)

    def test_serialization_failure_keeps_pre_emit_committed_digest(self):
        # Same fault and assertions; injection targets the new encoder boundary.
        original = P.canonical_fast
        def canonical(value, backend='python'):
            if isinstance(value, dict) and 'generator_exclusive_ns' in value:
                raise F01.Fault('serialize original')
            return original(value, backend)
        _, failure, _, _ = self.exercise(F01.ToyEndpoint(), patches=(patch.object(P, 'canonical_fast', canonical),))
        self.check_prefix(failure, 0)
        self.assertTrue(failure['current_attempt']['record_constructed'])
        self.assertTrue(failure['current_attempt']['candidate_ready'])
        self.assertIsNotNone(failure['candidate_trace_sha256'])
        self.assertEqual(failure['stream_prefix_before_failure_record']['failure']['stage'], 'SERIALIZATION')
        self.assertFalse(failure['current_attempt']['stream_emit_returned'])


class NativeStreamF01(PythonStreamF01):
    backend = 'orjson'


if __name__ == '__main__':
    unittest.main()
