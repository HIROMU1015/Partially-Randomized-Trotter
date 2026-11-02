"""Stub-only API/launch checks. No registered synthesis, matrices or budgets."""
import importlib.util
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp
from trottertracks.algorithm_codesign.g9_v2_launch import validate_binding, verify_launch
from trottertracks.algorithm_codesign.rte_reallocation import numeric
from trottertracks.algorithm_codesign.rte_reallocation.native import Angle

ROOT = Path(__file__).resolve().parents[3]
PREP = ROOT / 'artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10'
spec = importlib.util.spec_from_file_location(
    'g9_v2_boundary_test_runner', ROOT / 'scripts/tracks/algorithm_codesign/g9_p5_matched_native_v2.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
GRID_MODULE = importlib.import_module('pygridsynth.gridsynth')


class BackendSentinel(Exception):
    """Proves the argument adapter reaches a STUB, never an acquisition."""


class G9V2Boundary(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads((PREP / 'contract_v2.json').read_text())
        self.angle = Angle('atan', F(1, 3))  # Off registered domain; stub-only.
        self.source, self.head = 'a' * 40, 'b' * 40
        self.allowed = {'authorization.json', 'receipt.md'}
        self.auth = {'source_commit': self.source, 'status': 'APPROVED_FOR_ONE_G9_V2_RUN',
                     'science_execution_authorized': True, 'runs': 1, 'retries': 0,
                     'mandatory_STOP': True, 'contract_sha256': 'c' * 64,
                     'explicit_execution_instruction': 'Synthetic binding fixture only; never execute science.'}

    def binding(self, auth=None, **overrides):
        values = {'contract_hash': 'c' * 64, 'requested_source': self.source,
                  'head': self.head, 'parents': [self.source],
                  'changed': ['authorization.json'], 'dirty': False, 'allowed': self.allowed}
        values.update(overrides)
        return validate_binding(self.auth if auth is None else auth, **values)

    def test_adapter_preserves_contract_string_and_key(self):
        sentinel = object()
        with patch.object(runner, 'synthesize', return_value=sentinel) as stub:
            self.assertIs(runner.acquire_fixed_primitive(self.angle, self.contract), sentinel)
        args, kwargs = stub.call_args
        self.assertIs(args[0], self.angle)
        self.assertIsInstance(args[1], str)
        self.assertEqual(args[1], '1/1000000')
        self.assertEqual(F(args[1]), F(1, 10**6))
        self.assertEqual(numeric.synthesis_key(self.angle, args[1]),
                         numeric.synthesis_key(self.angle, F(args[1])))
        self.assertEqual(args[2], self.contract['synthesizer_options'])
        self.assertEqual(kwargs, {'max_characters': 20000})

    def test_real_numeric_adapter_reaches_stub_with_epsilon_over_four(self):
        # numeric.synthesize is real; acquisition and strict guard are blocked.
        with patch.object(GRID_MODULE, 'gridsynth_gates', side_effect=BackendSentinel) as stub, \
                patch.object(numeric, 'strict_guard', side_effect=AssertionError('guard forbidden')) as guard:
            with self.assertRaises(BackendSentinel):
                runner.acquire_fixed_primitive(self.angle, self.contract)
            self.assertEqual(stub.call_count, 1)
            args, kwargs = stub.call_args
            self.assertEqual(args[1], mp.mpf('1/1000000') / 4)
            self.assertIsNotNone(kwargs['cfg'])
            guard.assert_not_called()

    def test_old_fraction_failure_occurs_before_backend_stub(self):
        with patch.object(GRID_MODULE, 'gridsynth_gates', side_effect=AssertionError('backend forbidden')) as stub, \
                patch.object(numeric, 'strict_guard', side_effect=AssertionError('guard forbidden')) as guard:
            with self.assertRaisesRegex(TypeError, 'cannot create mpf from Fraction'):
                numeric.synthesize(self.angle, F(1, 10**6), self.contract['synthesizer_options'])
            stub.assert_not_called()
            guard.assert_not_called()

    def test_bad_epsilon_object_rejected_before_helper(self):
        c = dict(self.contract, primitive_error=F(1, 10**6))
        with patch.object(runner, 'synthesize', side_effect=AssertionError('helper forbidden')) as stub:
            with self.assertRaisesRegex(TypeError, 'contract string'):
                runner.acquire_fixed_primitive(self.angle, c)
            stub.assert_not_called()

    def test_phase_mode_stays_strict(self):
        self.assertIs(self.contract['synthesizer_options']['up_to_phase'], False)
        with patch.object(GRID_MODULE, 'gridsynth_gates', side_effect=AssertionError('backend forbidden')) as stub:
            with self.assertRaisesRegex(ValueError, 'strict phase'):
                numeric.synthesize(self.angle, '1/1000000', {'up_to_phase': True})
            stub.assert_not_called()

    def test_approved_binding_fixture(self):
        self.binding()

    def test_pending_authorization_rejected(self):
        auth = json.loads((PREP / 'authorization.json').read_text())
        with self.assertRaises(PermissionError):
            self.binding(auth)

    def test_v1_approval_does_not_authorize_v2(self):
        with self.assertRaises(PermissionError):
            self.binding(dict(self.auth, status='APPROVED_FOR_ONE_G9_RUN'))

    def test_instruction_required(self):
        with self.assertRaises(PermissionError):
            self.binding(dict(self.auth, explicit_execution_instruction=None))

    def test_contract_and_source_binding(self):
        with self.assertRaises(PermissionError):
            self.binding(contract_hash='d' * 64)
        with self.assertRaises(PermissionError):
            self.binding(requested_source='d' * 40)

    def test_direct_child_only_and_clean(self):
        for overrides in ({'parents': []}, {'parents': [self.source, 'c' * 40]},
                          {'head': self.source}, {'dirty': True}):
            with self.assertRaises(PermissionError):
                self.binding(**overrides)

    def test_authorization_only_diff(self):
        with self.assertRaises(PermissionError):
            self.binding(changed=['authorization.json', 'contract.json'])
        with self.assertRaises(PermissionError):
            self.binding(changed=[])

    def test_limits_cannot_be_relaxed_by_authorization(self):
        for changed in ({'runs': 2}, {'runs': True}, {'retries': 1},
                        {'science_execution_authorized': False}, {'mandatory_STOP': False}):
            with self.assertRaises(PermissionError):
                self.binding(dict(self.auth, **changed))

    def test_pending_launch_refuses_before_git_or_output_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'contract.json').write_text(json.dumps({'authorization_path': 'auth.json'}))
            (root / 'auth.json').write_text(json.dumps({'source_commit': None, 'science_execution_authorized': False}))
            with patch('trottertracks.algorithm_codesign.g9_v2_launch.git', side_effect=AssertionError('git forbidden')) as git:
                with self.assertRaisesRegex(PermissionError, 'preparation only'):
                    verify_launch(root, root / 'contract.json', self.source)
                git.assert_not_called()
            self.assertEqual({p.name for p in root.iterdir()}, {'auth.json', 'contract.json'})

    def test_pending_entrypoint_cannot_reach_marker_or_science(self):
        # Synthetic entrypoint unit test; no runner CLI or registered run.
        with patch.object(runner, 'verify_launch', side_effect=PermissionError('pending')), \
                patch.object(runner, 'consume_marker') as marker, \
                patch.object(runner, 'verify_runtime') as runtime, \
                patch.object(runner, 'synthesize') as backend, \
                patch.object(runner, 'generators') as generator, \
                patch.object(runner, 'row') as native:
            with self.assertRaises(PermissionError):
                runner.execute(self.source)
            for stub in (marker, runtime, backend, generator, native):
                stub.assert_not_called()

    def test_scientific_contract_fields_and_inventory_unchanged(self):
        check = json.loads((PREP / 'static_equivalence_v2.json').read_text())
        old = json.loads((ROOT / 'artifacts/track_b_g9_p5_native_preparation/2026-10-10/contract_v1.json').read_text())
        for name in check['unchanged_old_fields']:
            self.assertEqual(old[name], self.contract[name], name)
        self.assertEqual(self.contract['synthesis_inventory'],
                         'artifacts/track_b_g9_p5_native_preparation/2026-10-10/synthesis_inventory_v1.json')
        self.assertNotEqual(self.contract['result_directory'], old['result_directory'])

    def synthetic_launch_fixture(self, root):
        c = {'authorization_path': 'auth.json', 'optional_receipt_path': 'receipt.md',
             'source_manifest': 'manifest.json', 'protected_ledger': 'ledger.json',
             'append_only_paths': [], 'result_directory': 'out'}
        (root / 'contract.json').write_text(json.dumps(c))
        digest = hashlib.sha256((root / 'contract.json').read_bytes()).hexdigest()
        (root / 'auth.json').write_text(json.dumps(dict(self.auth, contract_sha256=digest)))
        (root / 'critical.txt').write_bytes(b'Synthetic critical bytes, not science input.')
        critical_hash = hashlib.sha256((root / 'critical.txt').read_bytes()).hexdigest()
        (root / 'manifest.json').write_text(json.dumps({'focused_tests_passed': True,
                                                      'sha256': {'critical.txt': critical_hash}}))
        (root / 'ledger.json').write_text('{}')
        def git_stub(unused_root, *args):
            if args == ('rev-parse', 'HEAD'):
                return self.head
            if args[:2] == ('diff', '--name-only'):
                return 'auth.json'
            if args[:2] == ('show', '-s'):
                return self.source
            if args[0] == 'status':
                return ''
            raise AssertionError(args)
        return git_stub

    def test_valid_synthetic_launch_gate_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            git_stub = self.synthetic_launch_fixture(root)
            with patch('trottertracks.algorithm_codesign.g9_v2_launch.git', side_effect=git_stub):
                c, auth, head, check = verify_launch(root, root / 'contract.json', self.source)
            self.assertEqual(head, self.head)
            self.assertEqual(check['violations'], [])
            self.assertFalse((root / 'out').exists())

    def test_consumed_synthetic_marker_blocks_even_approved_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            git_stub = self.synthetic_launch_fixture(root)
            (root / 'out').mkdir()
            marker = root / 'out/one_shot_consumed.json'
            marker.write_text('Synthetic consumed marker')
            before = marker.read_bytes()
            with patch('trottertracks.algorithm_codesign.g9_v2_launch.git', side_effect=git_stub):
                with self.assertRaisesRegex(FileExistsError, 'no retry'):
                    verify_launch(root, root / 'contract.json', self.source)
            self.assertEqual(marker.read_bytes(), before)

    def test_critical_source_mismatch_blocks_approved_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            git_stub = self.synthetic_launch_fixture(root)
            (root / 'critical.txt').write_text('Changed synthetic bytes')
            with patch('trottertracks.algorithm_codesign.g9_v2_launch.git', side_effect=git_stub):
                with self.assertRaisesRegex(PermissionError, 'critical source'):
                    verify_launch(root, root / 'contract.json', self.source)
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
