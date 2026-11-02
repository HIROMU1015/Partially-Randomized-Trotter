"""Off-domain API adapter and version protection; real synthesis never runs."""
import ast
import copy
from fractions import Fraction as F
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.algorithm_codesign.rte_reallocation import numeric
from trottertracks.algorithm_codesign.rte_reallocation.native import Angle
from trottertracks.algorithm_codesign.rte_robustness import launch, worker
from trottertracks.algorithm_codesign.rte_development.evidence import digest

C = json.loads((ROOT / launch.CONTRACT).read_text())
BACKEND_MODULE = importlib.import_module('pygridsynth.gridsynth')
OLD_SOURCE = 'c7d2c826433dcf3b1bb30c98fc05a3bf13970ae1'
SEQUENCE = 'THTHtSW'  # Invented sequence: no physical synthesis claim.


class PrecisionConnectionTests(unittest.TestCase):
    def setUp(self):
        numeric.configure(100)
        self.angle = Angle('atan', F(2, 1009))  # Not in the registered inventory.
        self.epsilon = F(1, 10**7)
        self.options = C['acquisition']['options']

    def acquire_mock(self, epsilon=None, bound=None, on_synthesis=None):
        eps = self.epsilon if epsilon is None else epsilon
        with patch.object(BACKEND_MODULE, 'gridsynth_gates', return_value=SEQUENCE) as backend, \
             patch.object(numeric, 'strict_guard', return_value=F(eps)/2 if bound is None else bound) as guard:
            saved = numeric.synthesize(self.angle, eps, self.options, on_synthesis=on_synthesis)
        return saved, backend, guard

    def test_real_helper_accepts_fraction_with_mock_backend(self):
        saved, backend, guard = self.acquire_mock()
        self.assertEqual(backend.call_count, 1)
        self.assertEqual(backend.call_args.args[1], numeric.mp.mpf(1)/40000000)
        self.assertIsInstance(backend.call_args.args[1], numeric.mp.mpf)
        self.assertEqual(guard.call_args.args, (SEQUENCE, self.angle))
        self.assertTrue(saved['error_pass'])

    def test_fraction_identity_is_not_changed_to_decimal(self):
        saved, _, _ = self.acquire_mock()
        self.assertEqual(saved['epsilon'], '1/10000000')
        self.assertEqual(saved['key'], numeric.synthesis_key(self.angle, self.epsilon))
        self.assertIs(numeric.validate_saved(saved, self.angle, self.epsilon), saved)

    def test_legacy_decimal_string_path_and_identity(self):
        saved, backend, _ = self.acquire_mock('1e-6')
        self.assertEqual(backend.call_args.args[1], numeric.mp.mpf('1e-6')/4)
        self.assertEqual(saved['epsilon'], '1e-6')
        self.assertIs(numeric.validate_saved(saved, self.angle, '1e-6'), saved)

    def test_legacy_fraction_string_path_and_identity(self):
        saved, backend, _ = self.acquire_mock('1/1000000')
        self.assertEqual(backend.call_args.args[1], numeric.mp.mpf('1/1000000')/4)
        self.assertEqual(saved['epsilon'], '1/1000000')
        self.assertIs(numeric.validate_saved(saved, self.angle, '1/1000000'), saved)

    def test_no_float_round_trip_for_small_fraction(self):
        epsilon = F(1, 2**180)
        _, backend, _ = self.acquire_mock(epsilon)
        self.assertEqual(backend.call_args.args[1], numeric.mp.mpf(1)/2**182)

    def test_original_threshold_inclusive_and_excess_rejected(self):
        saved, _, _ = self.acquire_mock(bound=self.epsilon)
        numeric.validate_saved(saved, self.angle, self.epsilon)
        excess, _, _ = self.acquire_mock(bound=self.epsilon+F(1, 10**40))
        self.assertFalse(excess['error_pass'])
        with self.assertRaises(PermissionError):
            numeric.validate_saved(excess, self.angle, self.epsilon)

    def test_options_phase_seed_and_dps_unchanged(self):
        _, backend, _ = self.acquire_mock()
        cfg = backend.call_args.kwargs['cfg']
        self.assertFalse(cfg.up_to_phase)
        self.assertEqual(cfg.seed, 0)
        self.assertEqual(cfg.dps, 100)

    def test_projective_option_rejected_before_backend_or_token(self):
        options = dict(self.options, up_to_phase=True)
        token = Mock()
        with patch.object(BACKEND_MODULE, 'gridsynth_gates') as backend:
            with self.assertRaises(ValueError):
                numeric.synthesize(self.angle, self.epsilon, options, on_synthesis=token)
        backend.assert_not_called(); token.assert_not_called()

    def test_token_immediately_precedes_one_backend_entry(self):
        events = []
        with patch.object(BACKEND_MODULE, 'gridsynth_gates',
                   side_effect=lambda *a, **k: events.append('entry') or SEQUENCE), \
             patch.object(numeric, 'strict_guard', return_value=F(0)):
            numeric.synthesize(self.angle, self.epsilon, self.options,
                               on_synthesis=lambda: events.append('token'))
        self.assertEqual(events, ['token', 'entry'])

    def test_bad_argument_does_not_reserve_backend_token(self):
        token = Mock()
        with patch.object(BACKEND_MODULE, 'gridsynth_gates') as backend:
            with self.assertRaises(ValueError):
                numeric.synthesize(self.angle, 'invalid precision', self.options, on_synthesis=token)
        backend.assert_not_called(); token.assert_not_called()

    def test_backend_failure_does_not_retry(self):
        token = Mock()
        with patch.object(BACKEND_MODULE, 'gridsynth_gates',
                   side_effect=ArithmeticError('invented backend failure')) as backend:
            with self.assertRaises(ArithmeticError):
                numeric.synthesize(self.angle, self.epsilon, self.options, on_synthesis=token)
        self.assertEqual(token.call_count, 1); self.assertEqual(backend.call_count, 1)

    def test_counts_sequence_hash_and_phase_record_unchanged(self):
        saved, _, _ = self.acquire_mock()
        self.assertEqual(saved['sequence_sha256'], hashlib.sha256(SEQUENCE.encode()).hexdigest())
        self.assertEqual(saved['T_count'], SEQUENCE.count('T')+SEQUENCE.count('t'))
        self.assertEqual(saved['Tdagger_count'], SEQUENCE.count('t'))
        self.assertEqual(saved['global_W_count'], 1)
        self.assertEqual(saved['one_qubit_count'], len(SEQUENCE)-1)

    def test_frozen_old_helper_reproduces_type_failure_off_domain(self):
        path = 'src/trottertracks/algorithm_codesign/rte_reallocation/numeric.py'
        old = subprocess.check_output(['git', 'show', OLD_SOURCE+':'+path], cwd=ROOT).decode()
        fn = next(n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef) and n.name == 'synthesize')
        namespace = dict(vars(numeric))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'frozen_off_domain_helper', 'exec'), namespace)
        token = Mock()
        with patch.object(BACKEND_MODULE, 'gridsynth_gates') as backend:
            with self.assertRaisesRegex(TypeError, 'cannot create mpf from Fraction'):
                namespace['synthesize'](self.angle, F(1, 1000003), self.options, on_synthesis=token)
        self.assertEqual(token.call_count, 1); backend.assert_not_called()

    def test_worker_real_helper_mock_backend_success_and_restart_protection(self):
        # Mock the backend boundary, not the helper whose adapter failed in v1.
        with tempfile.TemporaryDirectory() as t:
            d = Path(t); job_path = d/'key_000.json'; job_path.write_text('{}')
            job = dict(ratio='2/1009', key='invented', parent_pid=os.getpid())
            with patch.object(worker, 'verify_job', return_value=(C, job, d)), \
                 patch.object(worker.resource, 'setrlimit'), \
                 patch('trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch.verify_runtime'), \
                 patch.object(BACKEND_MODULE, 'gridsynth_gates', return_value=SEQUENCE) as backend, \
                 patch.object(numeric, 'strict_guard', return_value=F(0)):
                self.assertEqual(worker.execute(ROOT, job_path), 0)
                saved_bytes = job_path.with_suffix('.result.json').read_bytes()
                value = json.loads(saved_bytes)
                self.assertEqual(value['status'], 'KEY_COMPLETE')
                self.assertEqual(value['saved']['epsilon'], '1/10000000')
                self.assertEqual(backend.call_count, 1)
                with self.assertRaises(FileExistsError): worker.execute(ROOT, job_path)
                self.assertEqual(backend.call_count, 1)
                self.assertEqual(job_path.with_suffix('.result.json').read_bytes(), saved_bytes)


class V2IsolationTests(unittest.TestCase):
    def test_v1_authorization_cannot_authorize_v2(self):
        old = json.loads((ROOT/'artifacts/track_b_d3_source/2026-10-13/v1/authorization.json').read_text())
        self.assertTrue(old['science_execution_authorized'])
        with self.assertRaises(PermissionError):
            launch.validate_binding(old, digest(ROOT/launch.CONTRACT), OLD_SOURCE,
                                    'db584d9fb3f80c07ae55d5e1fe477bff2e386af2', [OLD_SOURCE],
                                    [C['execution']['authorization_path']], False, C['execution'])

    def test_old_consumed_marker_is_separate_and_new_namespace_absent(self):
        self.assertTrue((ROOT/'artifacts/track_b_d3_result/2026-10-13/v1/one_shot_consumed.json').exists())
        self.assertFalse((ROOT/C['execution']['result_directory']).exists())
        old = copy.deepcopy(C)
        old['execution']['result_directory'] = 'artifacts/track_b_d3_result/2026-10-13/v1'
        with self.assertRaises(PermissionError): launch.validate_contract(ROOT, old)

    def test_v1_science_projection_and_inventory_unchanged(self):
        old = json.loads((ROOT/'artifacts/track_b_d3_source/2026-10-13/v1/execution_contract_v1.json').read_text())
        for key in old:
            if key not in ('kind', 'next_authorized_work_now', 'execution'):
                self.assertEqual(C[key], old[key], key)

    def test_historical_blob_tampering_is_rejected(self):
        relative = 'invented.py'
        raw = b'# synthetic immutable source\n'
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for path, value in ((C['execution']['source_manifest'],
                                 dict(focused_tests_passed=True, critical_sha256={})),
                                (C['execution']['protected_ledger'],
                                 dict(sha256={}, historical_source_blobs={relative: dict(
                                     commit='1'*40, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())}))):
                p = root/path; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value))
            with patch.object(launch.subprocess, 'check_output', return_value=raw):
                launch.check_identities(root, C)
            with patch.object(launch.subprocess, 'check_output', return_value=raw+b'changed'):
                with self.assertRaises(PermissionError): launch.check_identities(root, C)


if __name__ == '__main__': unittest.main()
