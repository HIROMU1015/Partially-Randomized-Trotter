"""Focused off-domain semantics and synthetic launch/API tests; no live synthesis."""
import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
import numpy as np

from trottertracks.algorithm_codesign.g10_generator import (
    ClosedP5Tail, arm_names, static_support_bounds,
)
from trottertracks.algorithm_codesign.g10_reference import (
    reference_events, cts_events, exact_target, matrix_target,
)
from trottertracks.algorithm_codesign.g10_saved import (
    sqrt_bounds, exp_upper, log_bounds, affine_policy_lower,
)
from trottertracks.algorithm_codesign.g10_comparison import plan, rebudget_anchor
from trottertracks.algorithm_codesign.g10_launch import validate_binding, verify_launch
from trottertracks.algorithm_codesign.g7_generator import DeterministicBits, FullReturnGenerator
from trottertracks.algorithm_codesign.g9_native import native_ir, cost
from trottertracks.algorithm_codesign.g9_matrix import event_operator, circuit_error, pauli

ROOT = Path(__file__).resolve().parents[3]
PREP = 'artifacts/track_b_g10_degree_preparation/2026-10-10'
C = json.loads((ROOT/PREP/'contract_v1.json').read_text())
P, X = (F(2, 9), F(7, 9)), F(1, 3)  # off the registered p/x domain


class DegreeSemantics(unittest.TestCase):
    def test_P5_tail_m5_has_no_tail(self):
        g = ClosedP5Tail(P, X, 5)
        self.assertEqual(g.tail_indices, ())
        self.assertEqual(g.B, g.prefix.B)

    def test_P5_tail_only_six_seven_pair(self):
        g = ClosedP5Tail(P, X, 7)
        self.assertEqual([g.tail.groups[i].degree for i in g.tail_indices], [6])

    def test_invalid_even_degree(self):
        with self.assertRaises(ValueError):
            ClosedP5Tail(P, X, 6)

    def test_tail_proposal_exactly_normalized(self):
        self.assertEqual(sum(e['proposal'] for e in reference_events(ClosedP5Tail(P, X))), 1)

    def test_tail_corrected_weights_preserve_coefficients(self):
        self.assertTrue(all(e['proposal']*e['weight'] == e['coefficient']
                            for e in reference_events(ClosedP5Tail(P, X))))

    def test_prefix_coefficients_not_refitted(self):
        g = ClosedP5Tail(P, X)
        ref = {(e['word'], e['child']): e['coefficient'] for e in g.prefix.reference_events()}
        got = {(e['word'], e['child']): e['coefficient'] for e in reference_events(g)
               if len(e['raw_word']) < 6}
        self.assertEqual(got, ref)

    def test_full_P7_mean_with_prefix_plus_tail(self):
        events = list(reference_events(ClosedP5Tail(P, X)))
        mean = sum(float(e['coefficient'])*event_operator(e) for e in events)
        self.assertLess(np.linalg.norm(mean-matrix_target(P, X, 7), 2), 1e-12)

    def test_P7_full_local_mean(self):
        g = FullReturnGenerator(P, X, 7)
        mean = sum(float(e['coefficient'])*event_operator(e) for e in reference_events(g))
        self.assertLess(np.linalg.norm(mean-matrix_target(P, X, 7), 2), 1e-12)

    def test_source_matches_G9_CTS_at_m5_off_domain(self):
        from trottertracks.algorithm_codesign.g9_native import cts_events as old
        events, cert = cts_events(P, X, 5)
        original, oc = old(P, X)
        self.assertEqual(events, original)
        self.assertEqual(cert['rational_rotation_tangent'], oc['rational_rotation_tangent'])

    def test_CTS_general_P3_P7_operator_mean(self):
        for m in (3, 7):
            with self.subTest(m=m):
                events, cert = cts_events(P, X, m)
                mean = sum(float(e['coefficient'])*event_operator(e) for e in events)
                self.assertLess(np.linalg.norm(mean-matrix_target(P, X, m), 2), 1e-12)
                self.assertLessEqual(F(cert['coefficient_mean_error_upper']), 8*F(1, 10**12))

    def test_exact_Qsqrt2_target_independent_matrix(self):
        for m in (3, 7):
            target = sum(complex(float(v.a)+float(v.b)*np.sqrt(2))*(1j**phase)*pauli(axis)
                         for (axis, phase), v in exact_target(P, X, m).items())
            self.assertLess(np.linalg.norm(target-matrix_target(P, X, m), 2), 1e-12)

    def test_CTS_real_identity_and_zero_T_preserved(self):
        events, _ = cts_events(P, X, 7)
        real = [e for e in events if e['rotation_sign'] == 0]
        self.assertTrue(any(e['pauli'] == 'III' for e in real))
        self.assertTrue(all(cost(native_ir(e), {})['T'] == 0 for e in real))

    def test_P7_CTS_strict_phase_native_semantics_off_domain(self):
        for e in cts_events(P, X, 7)[0]:
            self.assertLess(circuit_error(e, native_ir(e), None), 1e-12)

    def test_degree6_controlled_word_phase(self):
        e = {'word': (0, 1, 2, 1, 0, 2), 'child': 1, 'ratio': F(1, 3), 'phase_i_power': 2}
        self.assertLess(circuit_error(e, native_ir(e), None), 1e-12)
        bad = deepcopy(e)
        bad['phase_i_power'] = 0
        self.assertGreater(np.linalg.norm(event_operator(bad)-event_operator(e), 2), 1)

    def test_production_has_no_reference_dependency(self):
        g = ClosedP5Tail(P, X)
        with patch.object(g.prefix, 'reference_events', side_effect=AssertionError('table forbidden')):
            with patch('trottertracks.algorithm_codesign.g7_reference.reference_events',
                       side_effect=AssertionError('table forbidden')):
                e = g.sample(DeterministicBits('G10-off-domain-fixture'))
        self.assertGreater(e['proposal'], 0)

    def test_full_production_without_reference(self):
        g = FullReturnGenerator(P, X, 7)
        with patch('trottertracks.algorithm_codesign.g7_reference.reference_events',
                   side_effect=AssertionError('table forbidden')):
            e = g.sample(DeterministicBits('G10-off-domain-fixture'))
        self.assertTrue(e is None or e['proposal'] > 0)

    def test_registered_row_shape_only(self):
        self.assertEqual([len(arm_names(m)) for m in (3, 5, 7)], [5, 6, 6])
        self.assertIn('closed_P5_tail', arm_names(7))
        with self.assertRaises(ValueError):
            arm_names(9)

    def test_static_binding_caps(self):
        bounds = static_support_bounds()
        self.assertEqual(len(bounds), C['rows'])
        self.assertLessEqual(sum(r['event_upper'] for r in bounds), C['caps']['event_bindings'])
        self.assertLessEqual(19+C['static_angle_upper_new'], C['cache_entries'])


class PolicyTests(unittest.TestCase):
    def test_union_failure_34_axes_and17_resource_rows(self):
        self.assertEqual(34*F(C['alpha_axis'])+17*F(C['resource_failure_per_row']), F(1, 20))
        self.assertLess(exp_upper(10), F(23000))
        self.assertGreater(log_bounds(17000)[1], 9)
        self.assertLess(log_bounds(17000)[1], 10)

    def test_digital_P7_moment_range_bound_off_domain(self):
        for g in (ClosedP5Tail(P, X), FullReturnGenerator(P, X, 7)):
            b = plan(C, 7, g=g)
            events = list(reference_events(g))
            self.assertLessEqual(sum(e['proposal']*e['weight']**2 for e in events), b['m2_upper'])
            self.assertLessEqual(max(e['weight'] for e in events), b['range_upper'])
            self.assertLessEqual(b['accepted_call_cap_two_axes'], b['hard_attempt_cap_two_axes'])

    def test_local_budget_does_not_use_reference_full_normalizer(self):
        b = plan(C, 7, g=FullReturnGenerator(P, X, 7))
        self.assertFalse(b['uses_signal_or_full_normalizer_for_local_budget'])
        self.assertNotIn('GO', b)

    def test_rebudget_anchor_is_saved_only(self):
        from trottertracks.algorithm_codesign.g9_comparison import plan as oldplan
        g = FullReturnGenerator(P, X, 5)
        b = oldplan(g)
        fake = {'budget': {k: str(v) if isinstance(v, F) else v for k, v in b.items()},
                'reference_acceptance': '3/4', 'per_trial_native_cost': {'T': '4', 'CX': '2', '1Q': '9'},
                'registered_worst_event_T': 12, 'events': ['opaque_saved_record'],
                'two_axis_expected_native_cost': {'T': '8'}, 'primary': True, 'arm': 'full_return'}
        with patch('trottertracks.algorithm_codesign.g9_matrix.circuit_error', side_effect=AssertionError()), \
             patch('trottertracks.algorithm_codesign.g9_native.native_ir', side_effect=AssertionError()):
            reb = rebudget_anchor(fake, C)
        self.assertEqual(fake['events'], reb['events'])
        self.assertEqual(reb['two_axis_expected_native_cost']['T'], 8*reb['budget']['N_per_axis'])
        self.assertNotEqual(reb['budget']['alpha_axis'], F(fake['budget']['alpha_axis']))
        self.assertEqual(fake['two_axis_expected_native_cost']['T'], '8')

    def test_sqrt_interval_exact_and_nonsquare(self):
        self.assertEqual(sqrt_bounds(4), (F(2), F(2)))
        lo, hi = sqrt_bounds(F(2, 3))
        self.assertLessEqual(lo*lo, F(2, 3))
        self.assertGreaterEqual(hi*hi, F(2, 3))

    def test_analytic_lower_zero_cost_retained(self):
        es = [{'event': {'coefficient': '1/3'}, 'cost': {'T': 0}},
              {'event': {'coefficient': '2/3'}, 'cost': {'T': 4}}]
        out = affine_policy_lower(es, '1/10', '1/20')
        self.assertEqual(out['zero_T_events_retained'], 1)
        self.assertEqual(out['weighted_root_lower'], F(4, 3))
        self.assertFalse(out['attained_optimum_or_executable_law_claim'])

    def test_analytic_lower_below_explicit_policy_for_h(self):
        es = [{'event': {'coefficient': '1/3'}, 'cost': {'T': 0}},
              {'event': {'coefficient': '2/3'}, 'cost': {'T': 4}}]
        lower = affine_policy_lower(es, '1/10', '1/20')
        ell = log_bounds(40)[1]
        for h in (0, 1, 5):
            m2 = F(1, 9)/F(1, 2)+F(4, 9)/F(1, 2)
            price = F(1, 2)*h+F(1, 2)*(4+h)
            cost_upper = 4*ell*m2*price/F(1, 10)**2
            self.assertLess(lower['intercept_lower']+h*lower['prep_slope_lower'], cost_upper)

    def test_negative_price_refused(self):
        with self.assertRaises(ValueError):
            affine_policy_lower([{'event': {'coefficient': '1'}, 'cost': {'T': -1}}], '1/10', '1/20')


class LaunchTests(unittest.TestCase):
    def auth(self):
        return {'status': 'APPROVED_FOR_ONE_G10_RUN', 'science_execution_authorized': True,
                'source_commit': 'a'*40, 'contract_sha256': 'digest',
                'explicit_execution_instruction': 'synthetic G10 one-shot instruction',
                'runs': 1, 'retries': 0, 'mandatory_STOP': True}

    def binding(self, auth=None, **changes):
        values = dict(auth=auth or self.auth(), contract_hash='digest', requested_source='a'*40,
                      head='b'*40, parents=['a'*40], changed=['authorization.json'],
                      dirty=False, allowed={'authorization.json', 'receipt.md'})
        values.update(changes)
        return validate_binding(**values)

    def test_direct_authorization_only_child(self):
        self.binding()

    def test_old_G9_authorization_refused(self):
        a = self.auth(); a['status'] = 'APPROVED_FOR_ONE_G9_V2_RUN'
        with self.assertRaises(PermissionError): self.binding(a)

    def test_source_HEAD_refused(self):
        with self.assertRaises(PermissionError): self.binding(head='a'*40)

    def test_merge_parent_refused(self):
        with self.assertRaises(PermissionError): self.binding(parents=['a'*40, 'c'*40])

    def test_dirty_or_source_change_refused(self):
        for updates in ({'dirty': True}, {'changed': ['authorization.json', 'source.py']}):
            with self.subTest(updates=updates), self.assertRaises(PermissionError): self.binding(**updates)

    def test_retry_or_boolean_run_refused(self):
        for k, v in (('retries', 1), ('runs', True), ('mandatory_STOP', False)):
            a = self.auth(); a[k] = v
            with self.subTest(k=k), self.assertRaises(PermissionError): self.binding(a)

    def test_wrong_contract_source_refused(self):
        for updates in ({'contract_hash': 'different'}, {'requested_source': 'c'*40}):
            with self.subTest(updates=updates), self.assertRaises(PermissionError): self.binding(**updates)

    def test_pending_refuses_before_git_or_protected_or_output(self):
        with patch('trottertracks.algorithm_codesign.g10_launch.git', side_effect=AssertionError('git early')), \
             patch('trottertracks.algorithm_codesign.g10_launch.protected_check', side_effect=AssertionError('data early')):
            with self.assertRaisesRegex(PermissionError, 'pending'):
                verify_launch(ROOT, ROOT/PREP/'contract_v1.json', 'a'*40)

    def gate_fixture(self, directory, remote='b'*40, marker=False, corrupt=False):
        import hashlib
        root = Path(directory)
        contract = {'authorization_path': 'authorization.json', 'optional_receipt_path': 'receipt.md',
                    'source_manifest': 'manifest.json', 'result_directory': 'new-result'}
        (root/'contract.json').write_text(json.dumps(contract))
        auth = self.auth()
        auth['contract_sha256'] = hashlib.sha256((root/'contract.json').read_bytes()).hexdigest()
        (root/'authorization.json').write_text(json.dumps(auth))
        (root/'critical.txt').write_text('synthetic fixture')
        h = hashlib.sha256((root/'critical.txt').read_bytes()).hexdigest()
        (root/'manifest.json').write_text(json.dumps({'focused_tests_passed': True,
                                                     'sha256': {'critical.txt': h}}))
        if corrupt: (root/'critical.txt').write_text('changed fixture')
        if marker:
            (root/'new-result').mkdir()
            (root/'new-result'/'one_shot_consumed.json').write_text('{}')
        def fake_git(_root, *args):
            if args[:2] == ('rev-parse', 'HEAD'): return 'b'*40
            if 'diff' in args: return 'authorization.json'
            if args[0] == 'show': return 'a'*40
            if args[0] == 'status': return ''
            if args[0] == 'branch': return 'synthetic-branch'
            if args[0] == 'ls-remote': return remote+'\trefs/heads/synthetic-branch'
            raise AssertionError(args)
        return root, fake_git

    def test_fresh_complete_launch_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root, fake = self.gate_fixture(directory)
            with patch('trottertracks.algorithm_codesign.g10_launch.git', fake), \
                 patch('trottertracks.algorithm_codesign.g10_launch.protected_check', return_value={'violations': []}):
                self.assertEqual(verify_launch(root, root/'contract.json', 'a'*40)[2], 'b'*40)
            self.assertFalse((root/'new-result').exists())

    def test_remote_mismatch_refused_without_science(self):
        with tempfile.TemporaryDirectory() as directory:
            root, fake = self.gate_fixture(directory, remote='c'*40)
            with patch('trottertracks.algorithm_codesign.g10_launch.git', fake):
                with self.assertRaisesRegex(PermissionError, 'remote'):
                    verify_launch(root, root/'contract.json', 'a'*40)

    def test_consumed_marker_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root, fake = self.gate_fixture(directory, marker=True)
            with patch('trottertracks.algorithm_codesign.g10_launch.git', fake), \
                 patch('trottertracks.algorithm_codesign.g10_launch.protected_check', return_value={'violations': []}):
                with self.assertRaises(FileExistsError):
                    verify_launch(root, root/'contract.json', 'a'*40)

    def test_critical_hash_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root, fake = self.gate_fixture(directory, corrupt=True)
            with patch('trottertracks.algorithm_codesign.g10_launch.git', fake):
                with self.assertRaisesRegex(PermissionError, 'critical'):
                    verify_launch(root, root/'contract.json', 'a'*40)

    def test_marker_exclusive_on_synthetic_directory(self):
        from trottertracks.algorithm_codesign.g7_launch import consume_marker
        with tempfile.TemporaryDirectory() as directory:
            marker = consume_marker(Path(directory)/'output', {'synthetic': True})
            original = marker.read_bytes()
            with self.assertRaises(FileExistsError):
                consume_marker(marker.parent, {'retry': True})
            self.assertEqual(marker.read_bytes(), original)

    def test_fixed_synthesis_type_boundary_stub_only(self):
        import importlib
        from trottertracks.algorithm_codesign.rte_reallocation.numeric import configure, synthesize
        from trottertracks.algorithm_codesign.rte_reallocation.native import Angle
        import mpmath as mp
        configure(100)
        seen = []
        class StubReached(Exception): pass
        def stub(theta, epsilon, cfg):
            seen.append(epsilon)
            raise StubReached()
        module = importlib.import_module('pygridsynth.gridsynth')
        with patch.object(module, 'gridsynth_gates', stub), \
             patch('trottertracks.algorithm_codesign.rte_reallocation.numeric.strict_guard', side_effect=AssertionError()):
            with self.assertRaises(StubReached):
                synthesize(Angle('atan', F(1, 3)), C['primitive_error'], C['synthesizer_options'])
        self.assertEqual(seen, [mp.mpf('1/1000000')/4])

    def test_future_runner_import_has_no_execution(self):
        path = ROOT/'scripts/tracks/algorithm_codesign/g10_degree_matched_native.py'
        spec = importlib.util.spec_from_file_location('future_g10', path)
        mod = importlib.util.module_from_spec(spec)
        with patch('trottertracks.algorithm_codesign.g10_launch.verify_launch', side_effect=AssertionError()):
            spec.loader.exec_module(mod)
        self.assertTrue(callable(mod.execute))


if __name__ == '__main__':
    unittest.main()
