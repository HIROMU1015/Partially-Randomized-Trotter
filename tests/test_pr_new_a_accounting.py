"""Artificial values only. Never opens private H4/H6/H8 scalar inputs."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import json
import math
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src/trotterlib'), str(ROOT / 'tests')]
import pr_new_a_accounting as a
import run_pr_new_a_accounting as r
import prepare_pr_new_a_accounting as prep
from test_pr_cost_integration import accounting, binding


def fixture():
    cells = [dict(id=f'B2_p{p}_q1_2nd_r4_K2', method='B2', formula='2nd', prefix=p,
                  q=1, r=4, K=2, R=4) for p in range(1, 17)]
    signals, tasks, pairs = [], [], []
    for ci, cell in enumerate(cells):
        signals.append(dict(cell=cell, precision_rows={str(e): {str(f): accounting() for f in (1., 10.)}
            for e in a.c.core.EPS}, empirical_noncancellation_precision_rows={str(e):
            {str(f): accounting() for f in (1., 10.)} for e in a.c.core.EPS}))
        for i in range(8):
            seed = 1000 + ci * 8 + i
            tasks.append(dict(cell=cell, replica=i, seed=seed))
            pairs.append(dict(cell_id=cell['id'], replica=i, seed=seed, event_digest='e' + str(seed),
                cohort='EXPLORATION_N8', binding=binding(), axes={axis:dict(seed=seed,
                event_digest='e' + str(seed), RZ=(10 + i + ci if axis == 'cosine' else 30 - i + ci))
                for axis in a.c.core.AXES}))
    old_ids = ['old' + str(i) for i in range(8)]
    old = [dict(cell_id=cid, method='B2' if cid in old_ids else 'B0', bias_mode=m,
        epsilon=e, factor=f, eligible=True, G_exact='500', G_SE=2 if cid in old_ids else None,
        shots=dict(cosine=3, sine=7), cost_status='FIXED_MODEL' if cid == 'det' else 'DIRECT_MEASURED')
        for cid in old_ids + ['det', 'det2'] for m, e, f in a.endpoints()]
    return dict(kind='ARTIFICIAL', system='H4', binding=binding(), cells=cells, signals=signals,
        signal_record_hashes={s['cell']['id']:a.c.digest(s) for s in signals}, tasks=tasks, pairs=pairs,
        old_PR_ids=old_ids, deterministic_sets=dict(old=['det'], strengthened=['det', 'det2']), old_rows=old)


def run(data):
    return a.account(data, a.Permit(synthetic=True))


class NewAccounting(unittest.TestCase):
    def test_exact_weighted_total_and_paired_SE(self):
        result = run(fixture())
        row = result['new_rows'][0]
        self.assertEqual(F(row['G_RZ_exact']), 226)
        self.assertAlmostEqual(row['G_RZ_SE'], math.sqrt(12))
        self.assertEqual(len(result['new_rows']), 320)
        self.assertEqual(len(result['comparisons']), 40)

    def test_old_sample_and_SE_are_preserved(self):
        data = fixture()
        result = run(data)
        self.assertEqual(result['unchanged_old_rows'], data['old_rows'])
        old = result['comparisons'][0]['old_saved']['best_available_PR']
        self.assertEqual(old['G_RZ_SE'], 2)
        self.assertEqual(old['cohort'], a.OLD_LABELS['H4'])
        self.assertNotIn('INDEPENDENT', old['cohort'])

    def test_missing_factor10_is_unresolved_without_fallback(self):
        data = fixture()
        for signal in data['signals']:
            for field in ('precision_rows', 'empirical_noncancellation_precision_rows'):
                for values in signal[field].values():
                    del values['10.0']
        data['signal_record_hashes'] = {s['cell']['id']:a.c.digest(s) for s in data['signals']}
        result = run(data)
        self.assertEqual(len(result['structural_audit']['missing_accounting_endpoints']), 160)
        for cmp in result['comparisons']:
            new = cmp['new_exploration']
            self.assertEqual(new['comparison_complete'], cmp['factor'] == 1.)
            if cmp['factor'] == 10.:
                self.assertIsNone(new['full_registered_set_PR_reduction_percent_exact'])
                self.assertIsNone(cmp['new_vs_old_PR_reduction_percent_exact'])

    def test_noncancellation_missing_never_uses_usual(self):
        data = fixture()
        for signal in data['signals']:
            del signal['empirical_noncancellation_precision_rows']
        data['signal_record_hashes'] = {s['cell']['id']:a.c.digest(s) for s in data['signals']}
        result = run(data)
        for row in result['new_rows']:
            if row['bias_mode'] == 'EMPIRICAL_NONCANCELLATION':
                self.assertIsNone(row['G_RZ_exact'])

    def test_single_missing_eligible_candidate_blocks_full_domain_claim(self):
        data = fixture()
        data['old_rows'][0]['eligible'] = None
        data['old_rows'][0]['G_exact'] = None
        cmp = run(data)['comparisons'][0]
        self.assertIsNone(cmp['old_saved']['full_registered_set_PR_reduction_percent_exact'])
        self.assertIsNone(cmp['new_vs_old_PR_reduction_percent_exact'])

    def test_all_ineligible_is_not_zero_cost(self):
        data = fixture()
        for s in data['signals']:
            s['precision_rows']['0.05']['1.0'] = accounting(False)
        data['signal_record_hashes'] = {s['cell']['id']:a.c.digest(s) for s in data['signals']}
        cmp = run(data)['comparisons'][0]
        self.assertIsNone(cmp['new_exploration']['best_available_PR'])

    def test_structure_audit_performs_no_numerical_function(self):
        with (patch.object(a.c.core, 'paired_statistics', side_effect=AssertionError('statistics')),
             patch.object(a.c.core, 'saved_shots', side_effect=AssertionError('shots')),
             patch.object(a.c.core, 'compare', side_effect=AssertionError('rank'))):
            self.assertEqual(a.audit_structure(fixture())['ranking'], 0)

    def test_input_is_immutable(self):
        data = fixture()
        before = deepcopy(data)
        run(data)
        self.assertEqual(data, before)

    def test_duplicate_endpoint_rejected(self):
        data = fixture()
        data['old_rows'].append(deepcopy(data['old_rows'][0]))
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_ENDPOINT'):
            run(data)

    def test_missing_registered_old_row_rejected(self):
        data = fixture()
        data['old_rows'].pop()
        with self.assertRaisesRegex(ValueError, 'COMPLETE_SAVED_D_DOMAIN'):
            run(data)

    def test_seed_replica_axis_and_cohort_corruption(self):
        for mutation in ('seed', 'replica', 'event', 'cohort', 'profile', 'cell', 'compiler'):
            data = fixture()
            pair = data['pairs'][0]
            if mutation == 'seed': pair['seed'] = -1
            if mutation == 'replica': pair['replica'] = 99
            if mutation == 'event': pair['axes']['sine']['event_digest'] = 'different'
            if mutation == 'cohort': pair['cohort'] = 'CONFIRMATION_N32'
            if mutation == 'profile': pair['binding']['profile'] = 'different'
            if mutation == 'cell': data['tasks'][0]['cell'] = data['cells'][1]
            if mutation == 'compiler': pair['axes']['sine']['compiler_hash'] = 'different'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                run(data)

    def test_shot_sum_conflict_stops(self):
        data = fixture()
        data['signals'][0]['precision_rows']['0.05']['1.0']['N_total'] = 11
        data['signal_record_hashes'][data['signals'][0]['cell']['id']] = a.c.digest(data['signals'][0])
        with self.assertRaisesRegex(ValueError, 'SHOTS'):
            run(data)

    def test_signal_hash_and_cell_parameters_cannot_change(self):
        for mutation in ('hash', 'parameters'):
            data = fixture()
            if mutation == 'hash': data['signals'][0]['extra'] = 1
            else: data['signals'][0]['cell'] = dict(data['cells'][0], R=5)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                run(data)

    def test_conditional_cost_never_accepted_as_n8(self):
        data = fixture()
        data['pairs'][0]['cohort'] = 'CONDITIONAL_N8'
        with self.assertRaisesRegex(ValueError, 'N8_PAIRED_ONLY'):
            run(data)

    def test_artificial_permit_cannot_process_real_data(self):
        data = fixture()
        data['kind'] = 'SAVED_REAL_VALUES'
        with self.assertRaisesRegex(ValueError, 'DATA_SCOPE_PERMIT'):
            run(data)
        with self.assertRaisesRegex(ValueError, 'SCOPED_GRANT'):
            a.Permit()

    def test_exact_tie_uses_id_and_cost_categories_stay_visible(self):
        result = run(fixture())
        cmp = result['comparisons'][0]['old_saved']
        self.assertEqual(cmp['best_available_PR']['cell_id'], 'old0')
        self.assertEqual(cmp['best_available_deterministic']['original_cost_status'], 'FIXED_MODEL')
        self.assertFalse(result['population_mean_certified'])
        self.assertIsNone(result['confidence_interval'])


class ExecutionBoundary(unittest.TestCase):
    def test_inherited_profile_id_is_not_promoted_to_file_SHA(self):
        value, kind = prep.profile_identity(dict(profile_sha256='legacy'), dict(profile='legacy'))
        self.assertEqual(kind, 'INHERITED_IDENTIFIER')
        self.assertNotIn('sha256', value)
        self.assertFalse(value['original_profile_file_bytes_verified'])

    def test_profile_mismatch_rejected_for_both_evidence_kinds(self):
        for native in (dict(profile_sha256='wrong'), dict(profile_reference=dict(sha256='wrong'))):
            with self.subTest(native=native), self.assertRaises(ValueError):
                prep.profile_identity(native, dict(profile='expected'))

    def test_selected_approval_receipt_must_match_contract(self):
        grant = dict(system='H4', scope='H4_NEW_A_TOTAL_COST', SOURCE='a'*40,
            technical_digest='expected', approved=True, approved_by_user=True,
            runtime_authorization=True, one_shot=True, retry=False,
            next_stage_authorized=False, approval_receipt='receipt', approval_receipt_sha256='hash')
        spec = dict(system='H4', scope=grant['scope'], SOURCE='a'*40, technical_digest='expected')
        receipt = dict(approved_by_user=True, SOURCE='a'*40, selected_contracts={grant['scope']:'other'})
        with patch.object(r.c, 'read', side_effect=[grant, spec, receipt]), patch.object(r.c, 'sha', return_value='hash'):
            with self.assertRaisesRegex(ValueError, 'USER_SELECTED_DIGEST'):
                r.execute('H4', Path('spec'), Path('grant'))

    def test_foreign_system_grant_cannot_authorize_selected_system(self):
        with self.assertRaisesRegex(ValueError, 'EXACT_NEW_SCOPE'):
            r.approve(dict(system='H6', scope='H6_NEW_A_TOTAL_COST'), 'H4')

    def test_technical_seal_tampering_is_rejected(self):
        spec = dict(schema=a.SCHEMA, technical_sealed=True, technical_digest='bad')
        with self.assertRaisesRegex(ValueError, 'TECHNICAL_DIGEST'):
            r.verify(spec)

    def test_invalid_grant_fails_before_spec_or_scientific_input(self):
        with patch.object(r.c, 'read', return_value=dict(system='H4', scope='H4_NEW_A_TOTAL_COST')) as read:
            with self.assertRaisesRegex(ValueError, 'EXPLICIT_APPROVAL'):
                r.execute('H4', Path('must-not-read'), Path('invalid'))
            self.assertEqual(read.call_count, 1)

    def test_old_grant_is_not_new_approval(self):
        with self.assertRaisesRegex(ValueError, 'EXACT_NEW_SCOPE'):
            r.approve(dict(system='H4', scope='H4_A_COST', approved=True), 'H4')

    def test_attempt_claim_is_persistent_and_stop_reason_is_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            r.c.claim_attempt(root/'lock', 'a'*40, 'synthetic')
            with self.assertRaises(FileExistsError): r.c.claim_attempt(root/'lock', 'a'*40, 'synthetic')
            r.c.first_stop(root, 'first')
            r.c.first_stop(root, 'later')
            self.assertEqual(r.c.read(root/'STOP.json')['reason'], 'first')

    def test_cli_invalid_grant_never_opens_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            grant = Path(directory)/'invalid.json'
            grant.write_text(json.dumps(dict(system='H4', scope='H4_NEW_A_TOTAL_COST', approved=False)))
            result = subprocess.run([sys.executable, '-P', '-B', str(ROOT/'scripts/run_pr_new_a_accounting.py'),
                '--system', 'H4', '--spec', str(Path(directory)/'missing'), '--run', '--grant', str(grant)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('EXPLICIT_APPROVAL', result.stderr)
            self.assertNotIn('No such file', result.stderr)


if __name__ == '__main__':
    unittest.main()
