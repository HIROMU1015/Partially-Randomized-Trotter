"""ARTIFICIAL endpoints and byte fixtures only; never read saved H8 values."""
from copy import deepcopy
import cmath
from datetime import datetime, timezone
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/trotterlib'))
sys.path.insert(0, str(ROOT/'scripts'))
import h8_signed_bias as n
import h8_signed_bias_contract as c
import run_h8_signed_bias_validation as runner

REGISTRY = json.loads((ROOT/'references/registration_v1.json').read_bytes())


class ArtificialSavedPermit:
    """Mock the saved-data permit for manufactured decoder bytes only."""
    def require(self, kind):
        if kind != 'SAVED_H8': raise AssertionError('ARTIFICIAL_DECODER_SCOPE')


def scalar(old=True):
    if old:
        return dict(C=.375, scalar_error_upper=0., exact_stored_scalar_numerator='3',
                    exact_stored_scalar_denominator='8', full_Fock_identity=True,
                    new_scientific_certificate=False)
    return dict(policy='EXACT_STORED_BINARY64_TAIL_IDENTITY', C=.375, numerator='3',
                denominator='8', scalar_error_upper=0., input_operator_error_certificate=False)


def fixture():
    ref = complex(.5, -.25)
    records = {}
    old_ids = {x['id'] for x in REGISTRY['old']}
    for cell in REGISTRY['old'] + REGISTRY['additional']:
        if cell['method'] == 'B0I':
            continue
        full = cell['method'] == 'B1'
        corrected, truncated = complex(.552, -.207), complex(.55, -.21)
        sig = dict(corrected=n.packed(corrected))
        if not full:
            sig['exact_truncated'] = n.packed(truncated)
        parts = {'PF': n.packed(corrected-ref)} if full else {
            'discard': n.packed(truncated-ref), 'PF': n.packed(corrected-truncated)}
        records[cell['id']] = dict(cell=deepcopy(cell), signals=sig, reference=n.packed(ref),
            independent_forward_signals=deepcopy(sig), error_decomposition=dict(
                signed_parts=parts, total_signed=n.packed(corrected-ref)))
    wrappers = []
    for cell in REGISTRY['old'] + REGISTRY['additional']:
        if cell['method'] != 'B0I':
            continue
        parent = records[cell['id'].replace('B0I_', 'B0_', 1)]
        phase = cmath.exp(-1j*.8*.375)
        sig = {k: n.packed(n.vector(v)*phase) for k, v in parent['signals'].items()}
        corrected, truncated = n.vector(sig['corrected']), n.vector(sig['exact_truncated'])
        row = dict(cell=deepcopy(cell), signals=sig, reference=n.packed(ref),
                   error_decomposition=dict(total_signed=n.packed(corrected-ref)))
        if cell['id'] in old_ids:
            wrappers.append(dict(prefix=cell['prefix'], q=cell['q'], scalar=scalar(), signal=row))
        else:
            row['scalar_receipt'] = scalar(False) if cell['prefix'] != 21 else dict(
                policy='REUSED_SAVED_P21_FULL_FOCK_SCALAR', receipt=scalar())
            row['error_decomposition']['signed_parts'] = dict(
                discard=n.packed(truncated-ref), PF=n.packed(corrected-truncated))
            row['independent_forward_signals'] = deepcopy(sig)
        records[cell['id']] = row
    return dict(kind='ARTIFICIAL', records=list(records.values()), old_B0I_wrappers=wrappers,
                full_reference=n.packed(ref), old_native_projection_cases=[(7, 1)],
                lineage_limitations=dict(shared_reference_and_scalar=True, certified=False))


def validate(data):
    return n.validate_domain(data, REGISTRY, c.Permit(synthetic=True, scope=n.SCOPE))


def selected(data, method='B0', old=True):
    ids = {r['id'] for r in (REGISTRY['old'] if old else REGISTRY['additional'])}
    return next(r for r in data['records'] if r['cell']['method'] == method and r['cell']['id'] in ids)


def artificial_input_files(root, mutate=lambda objects: None):
    """Minimal artificial copies of schemas, never copies of saved H8 numbers."""
    data = fixture()
    old_ids = {x['id'] for x in REGISTRY['old']}
    additional = [x for x in data['records'] if x['cell']['id'] not in old_ids]
    old = [x for x in data['records'] if x['cell']['id'] in old_ids and x['cell']['method'] != 'B0I']
    old += [dict(cell=dict(method='B2', id='ARTIFICIAL_PR_' + str(i))) for i in range(72)]
    full_ref = dict(reference=data['full_reference'], numerical_allowance_certified=False,
                    reference_agreement=dict(certified=False))
    encoded = lambda value: json.dumps(value, sort_keys=True, allow_nan=False).encode()
    old_bytes, decoded = encoded(dict(signals=old)), encoded(dict(status='COMPLETE_MANDATORY_STOP', signals=additional))
    zipped = gzip.compress(decoded)
    full_hash, old_hash = hashlib.sha256(encoded(full_ref)).hexdigest(), hashlib.sha256(old_bytes).hexdigest()
    workers = [dict(task_id='prefix_19' if i == 0 else 'ARTIFICIAL_' + str(i), exit_code=0, reaped=True,
                    original_result_sha256_present=i != 0, publication_result_sha256='a'*64) for i in range(35)]
    objects = dict(A_signals=zipped,
        A_export=dict(gzip_bytes=len(zipped), gzip_sha256=hashlib.sha256(zipped).hexdigest(),
                      original_json_bytes=len(decoded), original_json_sha256=hashlib.sha256(decoded).hexdigest()),
        A_binding=dict(SOURCE=c.A_SOURCE, provider_SOURCE=c.PROVIDER, profile_digest=c.PROFILE,
                       input_receipts=[dict(logical_name='full_H_reference', sha256=full_hash)]),
        A_terminal=dict(SOURCE=c.A_SOURCE, status='COMPLETE_MANDATORY_STOP'),
        A_workers=dict(workers=workers, all_reaped=True, original_result_hash_missing=['prefix_19'], original_receipts_modified=False),
        A_file_receipts=dict(files=[dict(path='signal_output/H8_signal_complete.json', bytes=len(decoded),
                                         sha256=hashlib.sha256(decoded).hexdigest())]),
        A_reference=dict(reference_receipt_sha256=full_hash), full_reference=full_ref,
        old_D0_contract=dict(binding=dict(source=c.PROVIDER), input_receipts=[dict(
            logical_name='signal_summary.json', bytes=len(old_bytes), sha256=old_hash)]),
        old_completion=dict(SOURCE=c.PROVIDER, status='COMPLETE_SAVED_RESULTS_VERIFIED'),
        old_signals=dict(signals=old), old_B0I=data['old_B0I_wrappers'],
        old_B0I_provenance=dict(execution_SOURCE=c.OLD_I_SOURCE, profile_digest=c.PROFILE),
        old_B0I_completion=dict(SOURCE=c.OLD_I_SOURCE, status='COMPLETE_MANDATORY_STOP'),
        old_B0I_correctness=[dict(prefix=p, q=q, status='PASS', certified=False)
                            for p, q in [(7, 1), (7, 2), (7, 4), (7, 8), (14, 1), (14, 2), (14, 4), (14, 8)]],
        registry=REGISTRY)
    mutate(objects)
    manifest = dict(schema='H8_BIAS_METADATA_AUDIT_INPUTS_V1', real_numerical_analysis_authorized=False, files=[])
    for role, value in objects.items():
        raw = value if type(value) is bytes else encoded(value)
        rel = role + ('.gz' if role == 'A_signals' else '.json')
        (root/rel).write_bytes(raw)
        manifest['files'].append(dict(role=role, snapshot_path=rel, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    return manifest


class NumericalTests(unittest.TestCase):
    def test_all_184_368_and_exactly_24_reconstructed_axes(self):
        result = validate(fixture())
        self.assertEqual(result['status'], 'PASS_EMPIRICAL_NUMERICAL_CONSISTENCY')
        self.assertEqual((result['candidate_count'], result['axis_count']), (184, 368))
        self.assertEqual(sum(r['reconstructed'] for r in result['rows']), 24)
        self.assertIsNone(result['rigorous_bias_guarantee'])
        self.assertEqual(result['noncancellation_shots'], 0)
        self.assertEqual(result['rankings'], 0)

    def test_old_overlay_parent_evidence_is_not_a_saved_independent_overlay(self):
        r = next(r for r in validate(fixture())['rows'] if r['reconstructed'])
        self.assertIn('NO_SEPARATE_OVERLAY_FORWARD', r['provenance']['forward_relation'])
        self.assertFalse(r['provenance']['reference_or_scalar_independence_certified'])

    def test_full_PF_has_PF_only(self):
        rows = [r for r in validate(fixture())['rows'] if r['cell']['method'] == 'B1']
        self.assertTrue(all(set(r['signed_parts']) == {'PF'} for r in rows))

    def test_saved_signed_component_mismatch_stops(self):
        d = fixture(); selected(d)['error_decomposition']['signed_parts']['discard']['real'] += .01
        result = validate(d)
        self.assertEqual(result['status'], 'STOP')
        self.assertIn('saved_discard_matches_endpoints', result['first_stop'])
        self.assertEqual(result['axis_count'], 368)

    def test_cancelling_wrong_parts_do_not_pass_by_closure_alone(self):
        d = fixture(); parts = selected(d)['error_decomposition']['signed_parts']
        parts['discard']['imag'] += .01; parts['PF']['imag'] -= .01
        self.assertEqual(validate(d)['status'], 'STOP')

    def test_saved_total_mismatch_and_first_reason_retained(self):
        d = fixture(); row = selected(d); row['error_decomposition']['total_signed']['real'] += .01
        row['error_decomposition']['signed_parts']['PF']['real'] += .02
        result = validate(d)
        self.assertIn('saved_total_matches_endpoints', result['first_stop'])

    def test_missing_one_axis_not_discarded(self):
        d = fixture(); row = selected(d); del row['signals']['exact_truncated']['imag']
        result = validate(d)
        own = [r for r in result['rows'] if r['cell']['id'] == row['cell']['id']]
        self.assertTrue(own[0]['status'].startswith('PASS'))
        self.assertEqual(own[1]['status'], 'FAIL_OR_MISSING')
        self.assertEqual(result['axis_count'], 368)

    def test_missing_duplicate_extra_candidate_fail_closed_full_domain(self):
        for mode in ('missing', 'duplicate', 'extra'):
            d = fixture()
            if mode == 'missing': d['records'].pop()
            elif mode == 'duplicate': d['records'].append(deepcopy(d['records'][0]))
            else:
                item = deepcopy(d['records'][0]); item['cell']['id'] = 'unregistered'; d['records'].append(item)
            with self.subTest(mode=mode):
                result = validate(d); self.assertEqual(result['status'], 'STOP')
                self.assertEqual(result['axis_count'], 368)

    def test_cell_and_time_changes_rejected(self):
        for key, value in [('prefix', 100), ('T', 1.)]:
            d = fixture(); selected(d)['cell'][key] = value
            with self.subTest(key=key): self.assertEqual(validate(d)['status'], 'STOP')

    def test_rotating_reference_is_rejected(self):
        d = fixture(); row = selected(d, 'B0I')
        row['reference'] = n.packed(n.vector(row['reference'])*cmath.exp(-1j*.8*.375))
        self.assertIn('full_H_reference_unrotated', validate(d)['first_stop'])

    def test_wrong_rotation_sign_rejected(self):
        d = fixture(); row = selected(d, 'B0I')
        parent = next(r for r in d['records'] if r['cell']['id'] == row['cell']['id'].replace('B0I_', 'B0_', 1))
        row['signals']['corrected'] = n.packed(n.vector(parent['signals']['corrected'])*cmath.exp(+1j*.8*.375))
        self.assertEqual(validate(d)['status'], 'STOP')

    def test_scalar_full_Fock_flag_and_exact_receipt_required(self):
        for key, value in [('full_Fock_identity', False), ('new_scientific_certificate', True),
                           ('exact_stored_scalar_denominator', '0'), ('C', .376)]:
            d = fixture(); d['old_B0I_wrappers'][0]['scalar'][key] = value
            with self.subTest(key=key): self.assertEqual(validate(d)['status'], 'STOP')

    def test_missing_or_duplicate_wrapper_rejected(self):
        for duplicate in (False, True):
            d = fixture()
            if duplicate: d['old_B0I_wrappers'].append(deepcopy(d['old_B0I_wrappers'][0]))
            else: d['old_B0I_wrappers'].pop()
            self.assertEqual(validate(d)['status'], 'STOP')

    def test_saved_forward_mismatch_rejected(self):
        d = fixture(); selected(d)['independent_forward_signals']['corrected']['real'] += .001
        self.assertEqual(validate(d)['status'], 'STOP')

    def test_new_overlay_forward_and_scalar_checked(self):
        d = fixture(); selected(d, 'B0I', old=False)['independent_forward_signals']['exact_truncated']['imag'] += .001
        self.assertEqual(validate(d)['status'], 'STOP')

    def test_nonfinite_bool_and_overflow_fail_closed(self):
        for value in (float('nan'), float('inf'), True, 10**400):
            d = fixture(); selected(d)['signals']['corrected']['real'] = value
            with self.subTest(type=type(value).__name__): self.assertEqual(validate(d)['status'], 'STOP')

    def test_no_mutation_and_repeat_is_deterministic(self):
        d = fixture(); before = deepcopy(d); a = validate(d); b = validate(d)
        self.assertEqual(a, b); self.assertEqual(d, before)

    def test_saved_kind_cannot_use_artificial_permit(self):
        d = fixture(); d['kind'] = 'SAVED_H8'
        with self.assertRaisesRegex(ValueError, 'DATA_KIND'): validate(d)

    def test_guard_interrupt_does_not_clear_first_reason(self):
        def guard(): raise InterruptedError('OWN_STOP')
        with self.assertRaisesRegex(InterruptedError, 'OWN_STOP'):
            n.validate_domain(fixture(), REGISTRY, c.Permit(synthetic=True, scope=n.SCOPE), guard)


class ContractTests(unittest.TestCase):
    def test_real_permit_requires_approved_grant(self):
        with self.assertRaisesRegex(ValueError, 'APPROVED_GRANT_PERMIT_REQUIRED'):
            c.Permit(synthetic=False, scope=n.SCOPE)
    @contextmanager
    def prepared(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root/'source'; source.mkdir()
            input_root = root/'inputs'; input_root.mkdir()
            manifest = dict(schema='ARTIFICIAL_MANIFEST_NO_SAVED_H8')
            files = []
            for path in c.SOURCE_PATHS:
                f = source/path; f.parent.mkdir(parents=True, exist_ok=True)
                raw = json.dumps(manifest).encode() if path.endswith('evidence_inputs_v1.json') else b'ARTIFICIAL_SOURCE'
                f.write_bytes(raw); files.append(dict(path=path, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            plan = dict(schema='H8_SIGNED_COMPONENT_EXECUTION_PLAN_V1', scope=n.SCOPE, SOURCE='a'*40,
                audit_commit=c.AUDIT, Phase_A_result=c.PHASE_A, saved_compiler_profile=c.PROFILE,
                tolerances=dict(n.TOLERANCES), comparison_rule='MATH_ISCLOSE_PER_REAL_IMAG_AXIS',
                science=dict(system='LINEAR_H8', separation_angstrom=1., basis='STO-3G', DF_rank=28,
                             modes=16, sector_dimension=4900, T=.8, candidates=184, axes=368),
                prohibited=dict(new_signal=0, compile=0, sampling=0, SCF_DF_state=0,
                    noncancellation_shots=0, PR_cost_comparison=0, ranking=0, fit=0),
                environment=c.environment(), source_files=files, input_manifest=manifest,
                resource_estimates=dict(new_output_bytes=4*2**20, new_output_inodes=4),
                input_manifest_sha256=c.digest(manifest))
            grant = dict(schema='H8_SIGNED_COMPONENT_GRANT_V1', scope=n.SCOPE, SOURCE='a'*40,
                approved=True, runtime_authorization=True, one_shot=True, automatic_retry=False,
                next_stage_authorized=False, authorization_evidence='ARTIFICIAL_UNITTEST_APPROVAL',
                input_root=str(input_root), output_directory=str(root/'output'),
                fresh_gate=dict(source_checked=True, environment_checked=True, resources_noninterference_checked=True,
                    checked_UTC=datetime.now(timezone.utc).isoformat()))
            grant['plan_sha256'] = c.digest(plan)
            yield root, source, plan, grant

    def test_authorization_binding_passes_artificial_contract_only(self):
        with self.prepared() as (_, source, plan, grant):
            permit, _, _ = c.authorize(plan, grant, source_root=source)
            self.assertFalse(permit.synthetic)

    def test_invalid_grant_rejected_before_any_input_or_output(self):
        with patch.object(c, 'load_saved', side_effect=AssertionError('MUST_NOT_READ')):
            with self.assertRaisesRegex(ValueError, 'EXPLICIT_NUMERICAL_EXECUTION'):
                runner.execute({}, dict(approved=False, runtime_authorization=False))

    def test_plan_digest_source_scope_and_fresh_gate_changes_rejected(self):
        for field in ('plan_sha256', 'SOURCE', 'scope', 'fresh_gate'):
            with self.prepared() as (_, source, plan, grant):
                grant[field] = {} if field == 'fresh_gate' else 'bad'
                with self.subTest(field=field), self.assertRaises((ValueError, KeyError)):
                    c.authorize(plan, grant, source_root=source)

    def test_tolerance_or_science_relaxation_rejected_even_with_rehashed_grant(self):
        with self.prepared() as (_, source, plan, grant):
            plan['tolerances']['abs_tol'] = 1e-6; grant['plan_sha256'] = c.digest(plan)
            with self.assertRaisesRegex(ValueError, 'FROZEN_SCIENCE_AND_TOLERANCES'):
                c.authorize(plan, grant, source_root=source)

    def test_source_byte_change_and_missing_closure_rejected(self):
        with self.prepared() as (_, source, plan, grant):
            (source/c.SOURCE_PATHS[0]).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'FIXED_INPUT'): c.authorize(plan, grant, source_root=source)
        with self.prepared() as (_, source, plan, grant):
            plan['source_files'].pop(); grant['plan_sha256'] = c.digest(plan)
            with self.assertRaisesRegex(ValueError, 'SOURCE_CLOSURE'): c.authorize(plan, grant, source_root=source)

    def test_profile_and_environment_mismatch_rejected(self):
        for key in ('saved_compiler_profile', 'environment'):
            with self.prepared() as (_, source, plan, grant):
                plan[key] = 'bad'; grant['plan_sha256'] = c.digest(plan)
                with self.subTest(key=key), self.assertRaises(ValueError): c.authorize(plan, grant, source_root=source)

    def test_stale_fresh_gate_is_rejected(self):
        with self.prepared() as (_, source, plan, grant):
            grant['fresh_gate']['checked_UTC'] = '2020-01-01T00:00:00+00:00'
            with self.assertRaisesRegex(ValueError, 'STALENESS'): c.authorize(plan, grant, source_root=source)

    def test_existing_output_symlink_and_path_escape_rejected(self):
        with self.prepared() as (root, source, plan, grant):
            Path(grant['output_directory']).mkdir()
            with self.assertRaisesRegex(ValueError, 'EXCLUSIVE'): c.authorize(plan, grant, source_root=source)
            link = root/'link'; link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, 'SYMLINK'): c.home_path(link)
            for path in ('/tmp/forbidden', str(root/'..'/'escape')):
                with self.assertRaises(ValueError): c.home_path(path)

    def test_strict_JSON_rejects_duplicates_nonfinite_and_overflow(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'):
            with self.assertRaises(ValueError): c.strict(raw)

    def test_byte_digest_changes_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'fixture'; path.write_bytes(b'ARTIFICIAL')
            receipt = dict(bytes=10, sha256=hashlib.sha256(b'ARTIFICIAL').hexdigest())
            self.assertEqual(c.read_verified(path, receipt), b'ARTIFICIAL')
            path.write_bytes(b'artificial')
            with self.assertRaisesRegex(ValueError, 'FIXED_INPUT'): c.read_verified(path, receipt)

    def test_one_shot_cleanup_and_no_extra_processes_artificial(self):
        with self.prepared() as (root, source, plan, grant):
            permit = c.Permit(synthetic=True, scope=n.SCOPE); output = root/'output'
            with patch.object(c, 'authorize', return_value=(permit, root/'inputs', output)), \
                 patch.object(c, 'ROOT', source), \
                 patch.object(c, 'load_saved', return_value=(fixture(), REGISTRY, [])):
                result = runner.execute(plan, grant)
                self.assertEqual(result['status'], 'PASS_EMPIRICAL_NUMERICAL_CONSISTENCY')
                self.assertEqual(result['child_processes_started'], 0)
                self.assertTrue(result['all_children_reaped'])
                with self.assertRaises(FileExistsError): runner.execute(plan, grant)

    def test_STOP_on_input_failure_keeps_lock_and_reason_no_retry(self):
        with self.prepared() as (root, _, plan, grant):
            output = root/'output'
            with patch.object(c, 'authorize', return_value=(c.Permit(synthetic=True, scope=n.SCOPE), root/'inputs', output)), \
                 patch.object(c, 'load_saved', side_effect=ValueError('FIRST_INPUT_FAILURE')):
                result = runner.execute(plan, grant)
                self.assertEqual(result['status'], 'STOP')
                self.assertIn('FIRST_INPUT_FAILURE', result['first_stop'])
                self.assertTrue((root/'output.one_shot_lock').is_file())
                self.assertFalse(result['automatic_retry'])

    def test_default_CLI_reads_no_saved_values(self):
        raw = subprocess.check_output([sys.executable, '-P', '-B', str(ROOT/'scripts/run_h8_signed_bias_validation.py')])
        result = json.loads(raw)
        self.assertEqual(result['saved_H8_numerical_analysis'], 0)
        self.assertFalse(result['runtime_authorization'])

    def test_artificial_schema_adapter_and_numeric_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); manifest = artificial_input_files(root)
            data, registry, receipts = c.load_saved(manifest, root, ArtificialSavedPermit())
            self.assertEqual(len(data['records']), 184)
            self.assertTrue(all(Path(r['path']).is_relative_to(root) for r in receipts))
            # All bytes above were manufactured by this test, not saved H8 data.
            data['kind'] = 'ARTIFICIAL'
            result = n.validate_domain(data, registry, c.Permit(synthetic=True, scope=n.SCOPE))
            self.assertEqual(result['status'], 'PASS_EMPIRICAL_NUMERICAL_CONSISTENCY')

    def test_artificial_adapter_rejects_wrong_source_profile_or_reference(self):
        for role, key in [('A_binding', 'SOURCE'), ('old_B0I_provenance', 'profile_digest'),
                          ('A_reference', 'reference_receipt_sha256')]:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                def mutate(objects): objects[role][key] = 'wrong'
                manifest = artificial_input_files(root, mutate)
                with self.subTest(role=role), self.assertRaises(ValueError):
                    c.load_saved(manifest, root, ArtificialSavedPermit())

    def test_original_runtime_hash_gap_cannot_be_backfilled(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def mutate(objects): objects['A_workers']['workers'][0]['original_result_sha256_present'] = True
            manifest = artificial_input_files(root, mutate)
            with self.assertRaisesRegex(ValueError, 'ORIGINAL_HASH_LIMITATION'):
                c.load_saved(manifest, root, ArtificialSavedPermit())

    def test_decompressed_bytes_binding_rejects_new_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def mutate(objects): objects['A_export']['original_json_sha256'] = 'bad'
            manifest = artificial_input_files(root, mutate)
            with self.assertRaisesRegex(ValueError, 'A_EXPORT_BYTES'):
                c.load_saved(manifest, root, ArtificialSavedPermit())

    def test_numeric_first_failure_survives_post_binding_failure(self):
        with self.prepared() as (root, _, plan, grant):
            output = root/'output'; data = fixture()
            selected(data)['error_decomposition']['signed_parts']['discard']['real'] += .01
            with patch.object(c, 'authorize', return_value=(c.Permit(synthetic=True, scope=n.SCOPE), root/'inputs', output)), \
                 patch.object(c, 'load_saved', return_value=(data, REGISTRY, [])), \
                 patch.object(c, 'read_verified', side_effect=ValueError('LATER_SOURCE_CHANGE')):
                result = runner.execute(plan, grant)
            self.assertIn('saved_discard_matches_endpoints', result['first_stop'])
            rows = json.loads((output/'axis_results.json').read_bytes())
            self.assertEqual(rows['axis_count'], 368)
            self.assertFalse(rows['post_execution_binding_verified'])


if __name__ == '__main__':
    unittest.main()
