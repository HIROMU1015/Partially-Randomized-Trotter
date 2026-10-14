"""Independent adversarial tests: fabricated scalars and two-mode mocks only.

No real grant is constructed, no saved scientific file is read, and no
scientific provider is imported or built. Tests use the existing artificial
fixture generator only as input scaffolding; the audit assertions are new.
"""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

_spec = importlib.util.spec_from_file_location(
    'independent_numeric_fixtures', Path(__file__).with_name('final_pre_review_fixtures_v1.py'))
f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f)
b = f.bridge


class IndependentNumericAuditTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ['TMPDIR']).resolve()
        if not scratch.is_relative_to('/home/AbeHiromu'):
            raise AssertionError('AUDIT_HOME_SCRATCH_REQUIRED')
        self.tmp = tempfile.TemporaryDirectory(prefix='independent-numeric-', dir=scratch)
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def h6_request(self):
        grant, record, fragments = f.h6_setup(self.base/'source', self.base/'output')
        prepared = f.h6_prepared(5)
        request = b.prepare(grant, prefix=5, q=1, binding=b.binding_for(grant, 5, 1),
            load_fragments=lambda _: fragments, load_prepared=lambda _: prepared,
            saved_record=record, scalar_input_error_upper=0., rotation_roundoff_upper=1e-12)
        provider = SimpleNamespace(
            build_deterministic_native=Mock(return_value='ARTIFICIAL_EVOLUTION'),
            build_native_hadamard_wrapper=Mock(return_value='ARTIFICIAL_WRAPPER'))
        return grant, request, provider, record

    def test_allowlisted_cross_source_paired_axes_are_rejected(self):
        binding, payloads, _ = f.synthetic_h8()
        binding['allowed_wrapper_sources'].append('9'*40)
        signals = f.d0.strict_json(payloads['signal_summary.json'])['signals']
        selection = f.d0.strict_json(payloads['selection_before_confirmation.json'])
        rows = f.d0.parse_wrappers(payloads['wrapper_metrics.csv'])
        for phase in ('exploration', 'confirmation'):
            changed = deepcopy(rows)
            row = next(r for r in changed if r['phase'] == phase and r['axis'] == 'sine')
            row['source'] = '9'*40
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                f.a.validate_inputs(signals, changed, selection, binding)

    def test_allowlisted_same_source_pair_remains_valid(self):
        binding, payloads, _ = f.synthetic_h8()
        binding['allowed_wrapper_sources'].append('9'*40)
        signals = f.d0.strict_json(payloads['signal_summary.json'])['signals']
        selection = f.d0.strict_json(payloads['selection_before_confirmation.json'])
        rows = f.d0.parse_wrappers(payloads['wrapper_metrics.csv'])
        first = rows[0]
        pair_key = (first['phase'], first['cell_id'], first['replica'], first['control'])
        for row in rows:
            if (row['phase'], row['cell_id'], row['replica'], row['control']) == pair_key:
                row['source'] = '9'*40
        self.assertTrue(f.a.validate_inputs(signals, rows, selection, binding)[
            'source_input_selection_bound'])

    def test_copied_request_cannot_repeat_native_build(self):
        grant, request, provider, _ = self.h6_request()
        copied = deepcopy(request)
        b.build(grant, request, provider)
        with self.assertRaises(b.g.Stop):
            b.build(grant, copied, provider)
        self.assertEqual(provider.build_deterministic_native.call_count, 1)

    def test_reset_built_flag_cannot_repeat_native_build(self):
        grant, request, provider, _ = self.h6_request()
        b.build(grant, request, provider)
        request['built'] = False
        with self.assertRaises(b.g.Stop):
            b.build(grant, request, provider)
        self.assertEqual(provider.build_deterministic_native.call_count, 1)

    def test_failed_provider_consumes_case_build_reservation(self):
        grant, request, provider, _ = self.h6_request()
        copied = deepcopy(request)
        provider.build_deterministic_native.side_effect = RuntimeError('ARTIFICIAL_FAILURE')
        with self.assertRaisesRegex(RuntimeError, 'ARTIFICIAL_FAILURE'):
            b.build(grant, request, provider)
        request['built'] = False
        for replay in (request, copied):
            with self.subTest(copied=replay is copied), self.assertRaises(b.g.Stop):
                b.build(grant, replay, provider)
        self.assertEqual(provider.build_deterministic_native.call_count, 1)

    def test_production_branch_rejects_name_only_unverified_provider(self):
        artificial_grant, request, provider, _ = self.h6_request()
        # This is a mock of the production branch, not an authorization receipt.
        mock_grant = SimpleNamespace(contract=deepcopy(artificial_grant.contract),
            synthetic=False, require=Mock())
        provider.__name__ = b.NATIVE_MODULE
        with self.assertRaises(b.g.Stop):
            b.build(mock_grant, request, provider)
        self.assertFalse(provider.build_deterministic_native.called)

    def test_mock_verified_provider_builds_and_rejects_same_name_substitution(self):
        artificial_grant, request, provider, _ = self.h6_request()
        source_name = 'src/trottertracks/resource_applicability/ax2b_native_df_v5.py'
        source_bytes = b'ARTIFICIAL_PROVIDER_SOURCE_ONLY\n'
        mock_grant = SimpleNamespace(contract=deepcopy(artificial_grant.contract),
            synthetic=False, require=Mock())
        mock_grant.contract['native_provider'] = dict(SOURCE='a'*40,
            source_hashes={source_name: b.g.sha(source_bytes)})
        checkout = self.base/'source'
        provider.__name__ = b.NATIVE_MODULE
        provider.__file__ = str(checkout/source_name)
        # Mock every input and import operation. No checkout source, Git child,
        # NumPy/Qiskit module, chemistry object, or native circuit is touched.
        with patch.object(b.g, 'read_bytes', return_value=source_bytes), \
                patch('subprocess.check_output', return_value=source_bytes) as git_mock, \
                patch('importlib.import_module', return_value=provider) as import_mock:
            verified = b.native_provider(mock_grant, checkout)
        self.assertIs(verified, provider)
        self.assertIs(mock_grant._verified_native_provider, provider)
        git_mock.assert_called_once()
        import_mock.assert_called_once_with(b.NATIVE_MODULE)
        impostor = SimpleNamespace(__name__=b.NATIVE_MODULE,
            build_deterministic_native=Mock(), build_native_hadamard_wrapper=Mock())
        with self.assertRaises(b.g.Stop):
            b.build(mock_grant, deepcopy(request), impostor)
        self.assertFalse(impostor.build_deterministic_native.called)
        b.build(mock_grant, request, provider)
        self.assertEqual(provider.build_deterministic_native.call_count, 1)

    def h8_pipeline_rejects_before_extension(self, mutate):
        binding, data, old_count = f.synthetic_h8()
        mutate(data)
        grant = SimpleNamespace(synthetic=True, require=Mock(), contract=dict(
            binding=binding, expected_old_precision_cases=944, expected_old_map_cases=old_count))
        artifacts = SimpleNamespace(write=Mock())
        original = f.driver.d0.accounting
        def account(record, epsilon, factor):
            self.assertNotEqual(epsilon, .0005, 'PHASE_B_REACHED_WITH_OLD_MISMATCH')
            return original(record, epsilon, factor)
        with patch.object(f.driver.d0, 'accounting', side_effect=account):
            with self.assertRaises((ValueError, f.g.Stop)):
                f.driver.pipeline(grant, data, f.FakeMonitor(grant), artifacts)
        self.assertFalse(artifacts.write.called)

    def test_final_confirmation_map_value_is_checked_before_new_epsilon(self):
        def mutate(data):
            old = f.d0.strict_json(data['confirmation_resource_map.json'])
            old['rows'][-1]['G_standard_error'] += 1
            data['confirmation_resource_map.json'] = json.dumps(old).encode()
        self.h8_pipeline_rejects_before_extension(mutate)

    def test_ineligible_old_precision_is_checked_before_new_epsilon(self):
        def mutate(data):
            summary = f.d0.strict_json(data['signal_summary.json'])
            record = next(s for s in summary['signals'] if
                not s['precision_rows']['0.05']['1.0']['cost_acquisition_eligible'])
            record['precision_rows']['0.001']['10.0']['boundary_guard'] = 'TAMPERED'
            data['signal_summary.json'] = json.dumps(summary).encode()
            selection = f.d0.strict_json(data['selection_before_confirmation.json'])
            selection['signal_digest'] = f.d0.digest(summary['signals'])
            data['selection_before_confirmation.json'] = json.dumps(selection).encode()
        # Rebind the fabricated selection digest so this reaches old equality,
        # not merely the input schema check.
        binding, data, old_count = f.synthetic_h8()
        mutate(data)
        binding['selection_digest'] = f.d0.digest(
            f.d0.strict_json(data['selection_before_confirmation.json']))
        grant = SimpleNamespace(synthetic=True, require=Mock(), contract=dict(
            binding=binding, expected_old_precision_cases=944, expected_old_map_cases=old_count))
        artifacts = SimpleNamespace(write=Mock())
        original = f.driver.d0.accounting
        def account(record, epsilon, factor):
            self.assertNotEqual(epsilon, .0005)
            return original(record, epsilon, factor)
        with patch.object(f.driver.d0, 'accounting', side_effect=account):
            with self.assertRaisesRegex(ValueError, 'ORIGINAL_PRECISION_ROWS_MISMATCH'):
                f.driver.pipeline(grant, data, f.FakeMonitor(grant), artifacts)
        self.assertFalse(artifacts.write.called)

    def test_entire_old_confirmation_map_is_required_before_new_epsilon(self):
        def mutate(data):
            old = f.d0.strict_json(data['confirmation_resource_map.json'])
            old['rows'].pop()
            data['confirmation_resource_map.json'] = json.dumps(old).encode()
        self.h8_pipeline_rejects_before_extension(mutate)

    def test_anticorrelated_axis_costs_keep_paired_covariance(self):
        pairs = [dict(cosine={'RZ': k}, sine={'RZ': 9-k}) for k in (1, 2, 4, 8)]
        result = f.d0.weighted_metric(pairs, {'cosine': 3, 'sine': 3}, 'RZ')
        self.assertEqual(result['G_point'], 27)
        self.assertEqual(result['G_standard_error'], 0.)
        self.assertEqual(result['G_exact'], {'numerator': '27', 'denominator': '1'})

    def test_signed_complex_identity_matches_full_fock_trace(self):
        # Explicit occupation-basis second quantization, independent of the
        # trace formula under audit. Basis dimension is four (two modes).
        matrix = ((1, 1j), (-1j, -2))
        op = [[0j for _ in range(4)] for _ in range(4)]
        for ket in range(4):
            for annihilate in range(2):
                if not (ket >> annihilate) & 1:
                    continue
                after = ket ^ (1 << annihilate)
                a_sign = (-1)**((ket & ((1 << annihilate)-1)).bit_count())
                for create in range(2):
                    if (after >> create) & 1:
                        continue
                    bra = after | (1 << create)
                    c_sign = (-1)**((after & ((1 << create)-1)).bit_count())
                    op[bra][ket] += matrix[create][annihilate]*a_sign*c_sign
        trace_square = sum(op[i][j]*op[j][i] for i in range(4) for j in range(4))
        independently_traced = -.5*trace_square.real/4
        self.assertEqual(trace_square.imag, 0)
        self.assertEqual(b.p.fragment_identity(matrix, -.5), independently_traced)
        self.assertEqual(b.exact_identity(matrix, -.5), Fraction(independently_traced))

    def test_saved_full_reference_is_not_rotated_or_certified(self):
        _, request, _, record = self.h6_request()
        self.assertEqual(request['signal']['reference'], record['reference'])
        self.assertFalse(request['signal']['allowance']['certified'])
        self.assertEqual(request['signal']['allowance']['kind'], 'EMPIRICAL')
        self.assertFalse(request['scalar_receipt']['new_scientific_certificate'])
        self.assertEqual(request['scalar_receipt']['declared_input_error_upper'], 0.)


if __name__ == '__main__':
    unittest.main()
