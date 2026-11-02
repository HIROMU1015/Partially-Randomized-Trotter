"""Synthetic metadata and arithmetic only. No saved H8 numbers are loaded."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/trotterlib'))
sys.path.insert(0, str(ROOT/'scripts'))
import h8_bias_evidence as e
import h8_bias_diagnostic_fixture as d
import audit_h8_bias_evidence as runner
REGISTRY = json.loads((ROOT/'references/registration_v1.json').read_text())


def record(cell, *, parts=True):
    full = cell['method'] == 'B1'
    components = {'PF': {'real': .01, 'imag': .002}}
    if not full: components['discard'] = {'real': -.005, 'imag': -.001}
    signal = dict(corrected=dict(real=.2, imag=.1))
    if not full: signal['exact_truncated'] = dict(real=.19, imag=.098)
    dec = dict(total_signed=dict(real=.005, imag=.001), complex_closure_discrepancy=0.)
    if parts: dec['signed_parts'] = components
    return dict(cell=deepcopy(cell), signals=signal, independent_forward_signals=deepcopy(signal),
                reference=dict(real=.195, imag=.099), error_decomposition=dec,
                allowance=dict(kind='EMPIRICAL', certified=False))


def fixture():
    records = []
    old_ids = {x['id'] for x in REGISTRY['old']}
    for cell in REGISTRY['old']+REGISTRY['additional']:
        parts = not (cell['id'] in old_ids and cell['method']=='B0I')
        row = record(cell, parts=parts)
        if not parts: row.pop('independent_forward_signals')
        records.append((row, dict(SOURCE='fixture', bytes_verified=True)))
    return records


def numeric():
    return dict(fixture_kind='ARTIFICIAL', signed_parts=dict(
        discard=dict(real=.02, imag=-.01), PF=dict(real=-.019, imag=.009)),
        total_signed=dict(real=.001, imag=-.001), allowance=dict(real=1e-9, imag=1e-9),
        allowance_kind='EMPIRICAL', allowance_certified=False)


class MetadataTests(unittest.TestCase):
    def test_184_and_368_metadata_partition(self):
        result=e.inventory(fixture(),REGISTRY)
        self.assertEqual(result['candidate_counts'],{e.SAVED:172,e.RECOVER:12})
        self.assertEqual(result['axis_counts'],{e.SAVED:344,e.RECOVER:24})
        self.assertIsNone(result['certified_bias_bound']);self.assertIsNone(result['full_184_empirical_minimum'])
        self.assertEqual(result['real_signed_component_recalculations'],0)

    def test_full_PF_single_component_is_not_missing_discard(self):
        cell=next(c for c in REGISTRY['old'] if c['method']=='B1')
        rows=e.inspect_record(record(cell),{})
        self.assertTrue(all(r['classification']==e.SAVED for r in rows))
        self.assertEqual(rows[0]['component_names'],['PF'])

    def test_metadata_audit_does_not_compute_component_closure(self):
        cell=REGISTRY['old'][0];r=record(cell)
        r['error_decomposition']['total_signed']['real']=123456.
        out=e.inspect_record(r,{})
        self.assertEqual(out[0]['classification'],e.SAVED)
        self.assertFalse(out[0]['closure_recomputed_this_audit'])
        self.assertFalse(out[0]['independent_component_numerical_agreement_verified_this_audit'])

    def test_missing_axis_is_classified_per_axis(self):
        r=record(REGISTRY['old'][0]);r['signals']['exact_truncated'].pop('imag')
        rows=e.inspect_record(r,{})
        self.assertEqual(rows[0]['classification'],e.SAVED)
        self.assertEqual(rows[1]['classification'],e.MISSING)

    def test_nonfinite_component_is_insufficient_not_zero(self):
        for value in (float('nan'),float('inf'),True):
            r=record(REGISTRY['old'][0]);r['error_decomposition']['signed_parts']['PF']['real']=value
            with self.subTest(value=value):self.assertEqual(e.inspect_record(r,{})[0]['classification'],e.MISSING)

    def test_missing_parts_with_endpoints_requires_future_approval(self):
        r=record(REGISTRY['old'][0],parts=False)
        self.assertEqual(e.inspect_record(r,{})[0]['classification'],e.RECOVER)

    def test_missing_total_is_not_reconstruction_authorization(self):
        r=record(REGISTRY['old'][0],parts=False);r['error_decomposition']['total_signed'].pop('real')
        self.assertEqual(e.inspect_record(r,{})[0]['classification'],e.MISSING)

    def test_unexpected_or_partial_components_not_silently_reconstructed(self):
        for components in ({'PF':dict(real=0.,imag=0.)},{'unknown':dict(real=0.,imag=0.)}):
            r=record(REGISTRY['old'][0]);r['error_decomposition']['signed_parts']=components
            self.assertEqual(e.inspect_record(r,{})[0]['classification'],e.MISSING)

    def test_duplicate_missing_extra_or_changed_cell_stops(self):
        for mode in ('duplicate','missing','changed','extra'):
            rows=fixture()
            if mode=='duplicate':rows[-1]=deepcopy(rows[0])
            elif mode=='missing':rows.pop()
            elif mode=='changed':rows[0][0]['cell']['prefix']=1234
            else:rows[0][0]['cell']['id']='not_registered'
            with self.subTest(mode=mode),self.assertRaises(ValueError):e.inventory(rows,REGISTRY)

    def test_rotated_forward_path_is_not_independent_scalar_certificate(self):
        cell=next(c for c in REGISTRY['additional'] if c['method']=='B0I')
        row=e.inspect_record(record(cell),{})[0]
        self.assertEqual(row['independent_path_relation'],'ROTATED_PARENT_FORWARD_ENDPOINTS_SHARED_SAVED_SCALAR')
        self.assertFalse(row['separate_independent_signed_components_saved'])
        self.assertFalse(row['rigorous_bias_certificate_present'])

    def test_record_without_forward_path_does_not_invent_it(self):
        r=record(REGISTRY['old'][0]);r.pop('independent_forward_signals')
        self.assertFalse(e.inspect_record(r,{})[0]['independent_forward_endpoints_present'])

    def test_strict_json_nonfinite_overflow_and_duplicate_stop(self):
        for value in (b'{"a":NaN}',b'{"a":1e999}',b'{"a":1,"a":2}'):
            with self.subTest(value=value),self.assertRaises(ValueError):runner.strict(value)

    def test_input_byte_change_or_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'fixture.json';p.write_bytes(b'{}')
            manifest=dict(schema='H8_BIAS_METADATA_AUDIT_INPUTS_V1',real_numerical_analysis_authorized=False,
                files=[dict(role='test',snapshot_path='fixture.json',bytes=2,sha256=runner.sha(p))])
            with patch.object(runner,'ROOT',root):
                runner.read_inputs(manifest)
                p.write_bytes(b'[]')
                with self.assertRaisesRegex(ValueError,'BYTES'):runner.read_inputs(manifest)
                p.unlink();(root/'target').write_bytes(b'{}');p.symlink_to(root/'target')
                with self.assertRaisesRegex(ValueError,'BYTES'):runner.read_inputs(manifest)

    def test_input_path_escape_and_duplicates_rejected(self):
        for path in ('../bad','/absolute'):
            manifest=dict(schema='H8_BIAS_METADATA_AUDIT_INPUTS_V1',real_numerical_analysis_authorized=False,
                files=[dict(role='test',snapshot_path=path)])
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'INPUT_PATH'):runner.read_inputs(manifest)

    def test_default_cli_is_plan_only(self):
        raw=subprocess.check_output([sys.executable,'-P','-B',str(ROOT/'scripts/audit_h8_bias_evidence.py')])
        value=json.loads(raw);self.assertEqual(value['new_bias_arithmetic'],0)
        self.assertFalse(value['runtime_authorization'])


class ArtificialDiagnosticTests(unittest.TestCase):
    def test_cancellation_and_noncancellation_differ_by_axis(self):
        result=d.artificial_envelopes(numeric())
        self.assertAlmostEqual(result['signed']['real'],.001)
        self.assertAlmostEqual(result['noncancellation']['real'],.039)
        self.assertAlmostEqual(result['noncancellation']['imag'],.019)
        self.assertFalse(result['certified']);self.assertIsNone(result['rigorous_bound'])

    def test_real_data_is_never_a_numerical_fixture(self):
        r=numeric();r.pop('fixture_kind')
        with self.assertRaisesRegex(ValueError,'ARTIFICIAL_DATA_ONLY'):d.artificial_envelopes(r)
        with self.assertRaisesRegex(ValueError,'ARTIFICIAL_DATA_ONLY'):d.artificial_shots(r,epsilon=.05,allowance_factor=1.)

    def test_future_closure_mismatch_stops(self):
        r=numeric();r['total_signed']['real']=.1
        with self.assertRaisesRegex(ValueError,'COMPONENT_CLOSURE'):d.artificial_envelopes(r)

    def test_missing_components_are_unresolved(self):
        r=numeric();r['signed_parts']=None;result=d.artificial_envelopes(r)
        self.assertEqual(result['status'],'UNRESOLVED_COMPONENTS');self.assertIsNone(result['noncancellation'])

    def test_full_PF_single_component_noncancellation_equals_signed(self):
        r=numeric();r['signed_parts']={'PF':deepcopy(r['total_signed'])}
        result=d.artificial_envelopes(r);self.assertEqual(result['signed'],result['noncancellation'])

    def test_no_axis_average_or_complex_norm_substitution(self):
        r=numeric();r['signed_parts']={'PF':dict(real=.001,imag=.1)};r['total_signed']=dict(real=.001,imag=.1)
        self.assertEqual(d.artificial_shots(r,epsilon=.05,allowance_factor=1.)['status'],
                         'INELIGIBLE_OR_UNRESOLVED_BOUNDARY')

    def test_empirical_shots_and_allowance_factors(self):
        r=numeric();r['signed_parts']={'PF':dict(real=.001,imag=.001)};r['total_signed']=dict(real=.001,imag=.001)
        a=d.artificial_shots(r,epsilon=.05,allowance_factor=1.)
        b=d.artificial_shots(r,epsilon=.05,allowance_factor=10.)
        self.assertEqual(a['status'],'ARTIFICIAL_EMPIRICAL_ELIGIBLE')
        self.assertGreaterEqual(b['shots']['real'],a['shots']['real'])
        self.assertFalse(a['population_guaranteed'])

    def test_factor_point_one_and_certificate_upgrade_rejected(self):
        with self.assertRaises(ValueError):d.artificial_shots(numeric(),epsilon=.05,allowance_factor=.1)
        r=numeric();r['allowance_certified']=True
        with self.assertRaisesRegex(ValueError,'CERTIFICATE_UPGRADE'):d.artificial_shots(r,epsilon=.05,allowance_factor=1.)

    def test_numerical_guard_and_unreliable_shots(self):
        r=numeric();r['signed_parts']={'PF':dict(real=0.,imag=0.)};r['total_signed']=dict(real=0.,imag=0.)
        r['allowance']={'real':.001,'imag':.001}
        self.assertEqual(d.artificial_shots(r,epsilon=.05,allowance_factor=1.)['status'],'NUMERICALLY_UNRESOLVED')

    def test_nonfinite_parts_or_overflow_rejected(self):
        for value in (float('nan'),float('inf'),True,1e308):
            r=numeric();r['signed_parts']['PF']['real']=value
            with self.subTest(value=value),self.assertRaises(ValueError):d.artificial_envelopes(r)

    def test_missing_domain_blocks_minimum_even_with_cheap_peer(self):
        rows=[dict(fixture_kind='ARTIFICIAL',id='cheap',status='ELIGIBLE',cost=1),
              dict(fixture_kind='ARTIFICIAL',id='missing',status='UNRESOLVED_COMPONENTS',cost=None)]
        out=d.artificial_full_domain(rows,['cheap','missing'])
        self.assertEqual(out['status'],'UNRESOLVED_FULL_DOMAIN');self.assertIsNone(out['best'])
        self.assertIsNone(out['PR_reduction'])

    def test_dropping_missing_candidate_is_rejected(self):
        rows=[dict(fixture_kind='ARTIFICIAL',id='cheap',status='ELIGIBLE',cost=1)]
        with self.assertRaisesRegex(ValueError,'FULL_REGISTERED_DOMAIN'):d.artificial_full_domain(rows,['cheap','missing'])

    def test_eligible_missing_cost_is_not_zero(self):
        rows=[dict(fixture_kind='ARTIFICIAL',id='candidate',status='ELIGIBLE',cost=None)]
        with self.assertRaisesRegex(ValueError,'COST_MISSING'):d.artificial_full_domain(rows,['candidate'])

    def test_ineligible_is_not_minimum_and_ties_are_lexical(self):
        rows=[dict(fixture_kind='ARTIFICIAL',id=x,status='ELIGIBLE',cost=5) for x in ('z','a')]
        rows.append(dict(fixture_kind='ARTIFICIAL',id='zero',status='INELIGIBLE',cost=None))
        self.assertEqual(d.artificial_full_domain(rows,['z','a','zero'])['best'],'a')

    def test_finite_RTE_and_outer_PF_are_separate_PR_diagnostic(self):
        r=numeric();r['signed_parts']={'outer_PF':r['signed_parts']['discard'],'finite_RTE':r['signed_parts']['PF']}
        self.assertEqual(d.artificial_envelopes(r)['status'],'EMPIRICAL_NONCANCELLATION_DIAGNOSTIC')


if __name__ == '__main__': unittest.main()
