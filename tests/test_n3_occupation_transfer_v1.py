"""Bounded synthetic suite (24 test methods); never reads transfer coefficients."""
import ast
import copy
from fractions import Fraction as F
from functools import lru_cache
import json
from pathlib import Path
import sys
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.representation_exploration import n3_occupation_transfer_v1 as gen
from trottertracks.representation_exploration.n3_occupation_transfer_check_v1 import Checker, check_saved


def fixture(kind='normal'):
    if kind == 'nu1':
        h = [[F(0), F(0), F(0)], [F(0), F(1), F(1, 10)], [F(0), F(1, 10), F(1)]]
        nu, density = 1, []
    elif kind == 'gain':
        h = [[F(-1), F(0), F(0), F(0)], [F(0), F(0), F(0), F(0)],
             [F(0), F(0), F(1), F(1, 10)], [F(0), F(0), F(1, 10), F(1, 2)]]
        nu, density = 2, [{'i': 0, 'j': 1, 'coefficient': F(1, 100)}]
    else:
        h = [[F(-1, 5), F(1, 50), F(0), F(1, 50)], [F(1, 50), F(0), F(0), F(0)],
             [F(0), F(0), F(1, 2), F(1, 100)], [F(1, 50), F(0), F(1, 100), F(2)]]
        nu, density = 2, [{'i': 0, 'j': 1, 'coefficient': F(1, 10)}]
    n = len(h)
    return gen.enc({'id': 'synthetic_' + kind, 'base': 'synthetic_' + kind, 'modes': n, 'particles': nu,
                    'active': list(range(n - 1)), 'external': n - 1, 'scale': F(1), 'amplitude': F(1, 50),
                    'direction': 'synthetic', 'h': h, 'density': density})


@lru_cache(maxsize=3)
def saved(kind='normal'):
    return gen.generate_case(fixture(kind))


def reject_record(record, kind='normal', message=None):
    c = Checker()
    try:
        c.audit_case(fixture(kind), record, 'synthetic')
    except (ValueError, KeyError, ZeroDivisionError):
        return
    raise AssertionError(message or 'tampered record accepted')


class SyntheticTests(unittest.TestCase):
    def test_01_coefficient_end_to_end(self):
        r = saved(); c = Checker(); c.audit_case(fixture(), r, 'synthetic')
        self.assertTrue(all(x['baseline_star'] for x in r['operational_rows']))
        self.assertEqual(r['truth_calls'], 0)

    def test_02_nu1_family_not_norm(self):
        r = saved('nu1'); Checker().audit_case(fixture('nu1'), r, 'synthetic')
        for row in r['raw_rows']:
            self.assertFalse(row['methods'][0]['accepted'])
            self.assertTrue(row['methods'][2]['accepted'])
            self.assertTrue(row['methods'][3]['accepted'])
        a = gen.Arithmetic(); A = [[F(0), F(0)], [F(0), F(1)]]; v = [F(0), F(1, 10)]
        D = F(251, 250); delta = F(1, 250)
        for k in (F(0), F(1, 100), F(1)):
            mu0 = a.neg(a.mul(delta, D))  # U=0, nu=1
            K = gen.plus(gen.scaled(A, k, a), gen.scaled(gen.outer(v, a), -1, a), a)
            old = gen.plus(gen.shifted(gen.scaled(A, k, a), mu0, a), gen.scaled(gen.outer(v, a), -1, a), a)
            self.assertEqual(old, gen.shifted(K, mu0, a))

    def test_03_norm_embedding_parameters(self):
        r = saved()
        f = next(f for f in r['operational_rows'][0]['fallback'] if f['certificate']['kind'] == 'norm')
        self.assertEqual(f['embedded_occ']['kappa'], [0, 1])
        self.assertEqual(f['embedded_occ']['spectral']['tau'], [0, 1])

    def test_04_old_embedding_parameters(self):
        r = saved('nu1')
        f = r['operational_rows'][-1]['fallback'][0]
        self.assertEqual(f['embedded_occ']['kappa'], f['certificate']['kappa'])
        self.assertEqual(f['embedded_occ']['spectral']['tau'], f['certificate']['mu'])

    def test_05_strong_bound_transport_preserves_witness(self):
        r = saved()
        moved = r['operational_rows'][-1]['baseline_certificates']
        self.assertTrue(any(w['origin']['row'] == 'c0' for w in moved))
        self.assertEqual(r['raw_rows'][0]['methods'][0]['attempts'][0]['binding']['row'], 'c0')
        Checker().audit_case(fixture(), r, 'synthetic')

    def test_06_affine_only_failure_candidate_reused(self):
        e = gen.Engine(); A = [[F(0), F(0)], [F(0), F(1)]]; v = [F(0), F(1, 10)]
        bind = gen.binding('synthetic', 'synthetic', F(0), F(1), F(1, 250), 'weak')
        with e.stage('old', 'OLD_DIRECT'):
            w = gen.old_candidate(F(1, 100), F(1, 1000), A, v, 1, F(0), F(1), F(1, 250), e, bind)
        # This test also checks a fixed PSD candidate whose weak affine margin fails.
        with e.stage('custom', 'OLD_DIRECT'):
            candidate = gen.old_candidate(F(1, 100), F(-1, 100), A, v, 1, F(0), F(1), F(1, 250), e, bind)
        self.assertTrue(candidate['PSD_valid']); self.assertFalse(candidate['accepted'])
        with e.stage('move', 'TRANSPORT'):
            moved = gen.transport(candidate, gen.binding('synthetic', 'synthetic', F(0), F(3), F(1, 250), 'strong'), A, v, 1, F(0), F(3), F(1, 250), e, {'row': 'weak', 'method': 'OLD_DIRECT', 'attempt': 0})
        self.assertTrue(moved['accepted']); self.assertEqual(moved['proof'], candidate['proof'])
        self.assertEqual(e.events[-1]['eig'], 0); self.assertEqual(e.events[-1]['psd'], 0)

    def test_07_psd_zero_pivot_and_rank_one(self):
        e = gen.Engine()
        with e.stage('psd', 'C0'):
            p = e.psd([[F(0), F(0)], [F(0), F(1)]])
            q = e.psd([[F(1), F(2)], [F(2), F(4)]])
        self.assertTrue(p['accepted'] and q['accepted'])
        c = Checker(); c.pool = e.proofs
        self.assertTrue(c.proof([[F(1), F(2)], [F(2), F(4)]], q['id'])[0])

    def test_08_zero_pivot_nonzero_row_rejected(self):
        e = gen.Engine()
        with e.stage('psd', 'C0'): p = e.psd([[F(0), F(1)], [F(1), F(0)]])
        self.assertFalse(p['accepted'])

    def test_09_tiny_negative_has_zero_tolerance(self):
        e = gen.Engine()
        with e.stage('psd', 'C0'): p = e.psd([[F(-1, 2 ** 400)]])
        self.assertFalse(p['accepted'])

    def test_10_schema_unknown_oracle(self):
        case = fixture(); case['true_gap'] = [1, 1]
        with self.assertRaises(ValueError): gen.generate_case(case)

    def test_11_bad_density_and_duplicate_pair(self):
        for kind in ('negative', 'duplicate'):
            case = fixture()
            if kind == 'negative': case['density'][0]['coefficient'] = [-1, 1]
            else:
                row = copy.deepcopy(case['density'][0]); row['i'], row['j'] = row['j'], row['i']; case['density'].append(row)
            with self.assertRaises(ValueError): gen.generate_case(case)

    def test_12_bad_partition_boolean_rational(self):
        for kind in ('partition', 'boolean'):
            case = fixture()
            if kind == 'partition': case['active'] = [2, 1, 0]
            else: case['h'][0][0] = [True, 1]
            with self.assertRaises(ValueError): gen.generate_case(case)

    def test_13_bad_source_binding(self):
        r = copy.deepcopy(saved()); r['source_fingerprint'] = 'wrong'
        reject_record(r)

    def test_14_false_upper_and_lower_rejected(self):
        for kind in ('U', 'c'):
            r = copy.deepcopy(saved())
            if kind == 'U': r['upper']['U'] = [-10, 1]
            else: r['bounds']['c_star'] = [10, 1]
            reject_record(r)

    def test_15_false_spectral_Z_and_margin_rejected(self):
        for kind in ('Z', 'margin'):
            r = copy.deepcopy(saved())
            w = r['raw_rows'][-1]['methods'][-1]['witness']
            if kind == 'Z': w['spectral']['Z'][0][0] = [-1, 1]
            else: w['affine_margin'] = [100, 1]
            reject_record(r)

    def test_16_false_psd_and_transport_rejected(self):
        for kind in ('proof', 'transport'):
            r = copy.deepcopy(saved())
            if kind == 'proof': next(iter(r['proof_pool'].values()))['accepted'] = False
            else: r['operational_rows'][-1]['baseline_certificates'][0]['binding']['c'] = [100, 1]
            reject_record(r)

    def test_17_rational_operation_cap(self):
        a = gen.Arithmetic(8)
        with self.assertRaises(gen.Unresolved): a.mul(F(15, 16), F(15, 16))
        r = gen.generate_case(fixture(), bit_cap=3)
        self.assertFalse(r['scientific_complete'])
        self.assertIn('ARITHMETIC_CAP', r['case_unresolved_reason'])
        with patch.object(gen.Engine, 'spectral', side_effect=gen.Unresolved('UNRESOLVED_SYNTHETIC_NUMERIC_FAILURE')):
            missing = gen.generate_case(fixture())
        Checker().audit_case(fixture(), missing, 'synthetic')
        self.assertIsNone(missing['raw_rows'][1]['c'])
        self.assertFalse(missing['operational_rows'][1]['occ_specific_gain'])

    def test_18_stage_reservation_not_shared_exhaustion(self):
        e = gen.Engine(cached=False)
        with e.stage('empty', 'NORM'):
            with self.assertRaises(gen.Unresolved): e.numeric([[1.]])
        with e.stage('old', 'OLD_DIRECT'):
            e.numeric([[1.]])
        self.assertEqual(e.counts['eig'], 1)

    def test_19_worst_case_budget_arithmetic(self):
        eig = 3 + 3 * (128 + 138) + 16
        psd = 35 + 3 * (5 + 10 + 120) + 2 + 6
        self.assertEqual((eig, psd), (817, 448))
        self.assertLessEqual(eig, gen.MAX_EIG); self.assertLessEqual(psd, gen.MAX_PSD)
        self.assertEqual(3 * 10 * 6, 180); self.assertEqual(2 * 6, 12)
        e = gen.Engine()
        with e.stage('m9', 'C0'):
            p = e.psd(gen.identity(9)); e.numeric(gen.identity(9))
        c = Checker(); c.pool = e.proofs
        self.assertTrue(c.proof(gen.identity(9), p['id'])[0])

    def test_20_cache_reference_semantics(self):
        reference = gen.generate_case(fixture(), cached=False)
        optimized = saved()
        def decision(r):
            return [[(m['method'], m['accepted'], m['witness']) for m in row['methods']] for row in r['raw_rows']]
        self.assertEqual(decision(reference), decision(optimized))
        Checker().audit_case(fixture(), reference, 'synthetic')
        self.assertGreater(optimized['resources']['counts']['cache_hits'], 0)

    def test_21_separator_after_primary_seal(self):
        r = saved('gain'); seal = gen.digest([r])
        s = gen.secondary_separator(fixture('gain'), r, seal)
        self.assertTrue(r['operational_rows'][-1]['occ_specific_gain'])
        self.assertEqual(s['status'], 'EXACT_SEPARATOR')
        c = Checker(); c.audit_case(fixture('gain'), r, 'synthetic')
        self.assertTrue(c.secondary(fixture('gain'), r, s, seal))

    def test_22_separator_both_endpoints_and_nonzero(self):
        r = saved('gain'); seal = gen.digest([r]); original = gen.secondary_separator(fixture('gain'), r, seal)
        for kind in ('endpoint', 'zero', 'seal'):
            s = copy.deepcopy(original)
            if kind == 'endpoint': s['separator']['endpoint_traces'][1] = [1, 1]
            elif kind == 'zero': s['separator']['trace'] = [0, 1]
            else: s['primary_seal'] = 'wrong'
            c = Checker(); c.audit_case(fixture('gain'), r, 'synthetic')
            with self.assertRaises(ValueError): c.secondary(fixture('gain'), r, s, seal)

    def test_23_gain_zero_one_two_bases(self):
        self.assertEqual(gen.classify([saved()])['status'], 'NO_ADDITIONAL_ACCEPTANCE_OBSERVED')
        gain = saved('gain')
        self.assertEqual(gen.classify([gain])['status'], 'SINGLE_BASE_GAIN_GATE_NOT_MET')
        second = copy.deepcopy(gain); second['base'] = 'synthetic_second'
        self.assertEqual(gen.classify([gain, second])['status'], 'FINITE_PIPELINE_TRANSFER_GAIN')
        self.assertFalse(gen.classify([gain, second])['family_separation'])

    def test_24_independent_checker_and_full_saved_packet(self):
        tree = ast.parse((ROOT / 'src/trottertracks/representation_exploration/n3_occupation_transfer_check_v1.py').read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any(name and ('numpy' in name or 'scipy' in name or 'n3_occupation_transfer_v1' in name) for name in imports))
        record = saved('gain'); seal = gen.digest([record]); extra = gen.secondary_separator(fixture('gain'), record, seal)
        report = check_saved({'cases': [fixture('gain')]}, [record], [extra], 'synthetic', seal)
        self.assertEqual(report['status'], 'INDEPENDENT_SAVED_WITNESSES_PASS')
        self.assertEqual(report['numerical_eigensolver_calls'], 0)
        # Explicit authorization gate must reject before plan/input loading.
        blocked = subprocess.run([sys.executable, str(ROOT / 'scripts/run_n3_occupation_transfer_v1.py'), '--run'], capture_output=True, text=True)
        self.assertEqual(blocked.returncode, 2)
        self.assertIn('explicit authorization required', blocked.stderr)
        sys.path.insert(0, str(ROOT / 'scripts'))
        import run_n3_occupation_transfer_v1 as runner
        plan = {'source_commit': 'synthetic', 'source_fingerprint': 'synthetic', 'input_sha256': 'synthetic'}
        invalid = {'scope': 'n3_occupation_transfer_v1_one_batch', 'scientific_batch_authorized': False,
                   'source_commit': 'synthetic', 'source_fingerprint': 'synthetic', 'plan_sha256': 'synthetic',
                   'input_sha256': 'synthetic', 'scientific_attempts': 1, 'scientific_retries': 0,
                   'next_scientific_stage_authorized': False, 'central_theme': None,
                   'execution_branch': 'n3-occupation-transfer-v1-execution-20261012', 'output_path': runner.RUN_PATH}
        with self.assertRaises(ValueError): runner.validate_authorization(invalid, plan, 'synthetic')
        watch = runner.MemoryWatch(); watch.start()
        try:
            parallel = runner.primary_pool([fixture(), fixture('nu1')], 'synthetic', watch=watch)
        finally: watch.stop()
        self.assertFalse(watch.violation)
        for case, value in zip((fixture(), fixture('nu1')), parallel): Checker().audit_case(case, value, 'synthetic')
        self.assertGreater(watch.peak, 0)


if __name__ == '__main__': unittest.main(verbosity=2)
