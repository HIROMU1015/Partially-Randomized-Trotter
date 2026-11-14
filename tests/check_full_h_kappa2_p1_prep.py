"""P-1 parent-side checks: fixed row/seeds, pinned SOURCE, C-guard sidecar format.

Metadata and fictional price values only. The circuit checks run in their own
process (provider root first), exactly like the real child.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_full_h_kappa2_p1 as P1  # noqa: E402
import revalidate_full_h_kappa2_p1_run02 as R  # noqa: E402
from trottertracks import price_implementation_guard as G  # noqa: E402

OLD = 'CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424'


class Tests(unittest.TestCase):
    def test_fixed_row_seeds_and_pinned_sources(self):
        P1.source_checks()
        prepared, req, row, seeds = P1.plan_row()
        self.assertEqual((row['q'], row['R_outer'], row['s'], row['K'], row['T'], row['prefix']), (8, 71, 568, 2, .8, 0))
        self.assertEqual(row['id'], 'FULL_H_4de6130f0d3a66e5d695')
        self.assertEqual(len(set(seeds)), 8)

    def test_price_sidecar_format_passes_unchanged_c_guard(self):
        prepared, req, row, seeds = P1.plan_row()
        trial = dict(scalar=-0.33147781341681615, prices={a: dict(rz_count=100 + i) for i, a in enumerate(('cosine', 'sine'))})
        prices, sides, proofs = P1.price_bindings(ROOT, prepared, trial, price_source=dict(fixture='ARTIFICIAL'),
            evidence_commit='a' * 40, evidence_path='ARTIFICIAL_IMPLEMENTATION_EVIDENCE.json')
        self.assertEqual([p['axis'] for p in prices], ['X', 'Y'])
        self.assertEqual({(p['classification'], p['expectation'], p['n_paths'], p['law_expectation'], p['record_schema'])
                          for p in prices}, {('DIRECT_COMPILED_FINITE_SAMPLE', 'FINITE_COHORT_ONLY', 1, False, P1.RECORD_SCHEMA)})
        self.assertEqual({p['diagnostic']['pilot_role'] for p in prices}, {'DIRECT_SINGLE_PATH_WIRING'})
        config = prepared['plan']['configuration']
        cohorts = {(config['rounds'][P1.M]['candidate']['id'], a): P1.COHORT for a in ('X', 'Y')}
        policy = R.policy(config)
        # This tree's guard and C's fixed guard (c628e9e8 blob, executed unchanged).
        for guard in (G, R.c_guard()):
            with patch.object(guard, '_document', return_value=proofs):
                out = guard.check(ROOT, config, prices, cohorts=cohorts, bindings=sides, policy=policy)
            self.assertEqual(len(out['checked']), 2)
            bad, bad_proofs = deepcopy(sides), deepcopy(proofs)
            for s in bad:
                s['implementation']['rte_rotation_lowering'] = OLD
            for p in bad_proofs.values():
                p['implementation']['rte_rotation_lowering'] = OLD
            with patch.object(guard, '_document', return_value=bad_proofs), self.assertRaises(ValueError):
                guard.check(ROOT, config, prices, cohorts=cohorts, bindings=bad, policy=policy)
            wrong = deepcopy(sides); wrong[0]['implementation']['resolved_scalar'] = 0.
            with patch.object(guard, '_document', return_value=proofs), self.assertRaises(ValueError):
                guard.check(ROOT, config, prices, cohorts=cohorts, bindings=wrong, policy=policy)
            stale = deepcopy(proofs); stale['3X']['price_source'] = dict(fixture='OTHER')
            with patch.object(guard, '_document', return_value=stale), self.assertRaises(ValueError):
                guard.check(ROOT, config, prices, cohorts=cohorts, bindings=sides, policy=policy)
        self.assertEqual(R.c_guard().__file__, R.C_GUARD['commit'] + ':' + R.C_GUARD['path'])

    def test_child_import_preflight_and_failed_import_control(self):
        report = P1.child_import_preflight()  # import-only + saved-input SHA; no restore or science
        self.assertEqual((report['status'], report['failed_imports'], report['missing_source_paths']), ('PASS', {}, []))
        self.assertTrue(all(report['input_checks'].values()))
        code = ('import json, sys; sys.path.insert(0, sys.argv[1]); import run_full_h_kappa2_p1 as P1; '
                'print(json.dumps(P1.import_preflight(P1.CHILD_IMPORTS + ("h46_absent_module_for_check",))))')
        out = subprocess.run([sys.executable, '-B', '-c', code, str(ROOT / 'scripts')], cwd=ROOT, capture_output=True, text=True)
        bad = json.loads(out.stdout)
        self.assertEqual(bad['status'], 'FAIL')
        self.assertEqual(list(bad['failed_imports']), ['h46_absent_module_for_check'])

    def test_circuit_checks_in_provider_root_process(self):
        out = subprocess.run([sys.executable, '-B', '-W', 'ignore', str(ROOT / 'tests/check_full_h_kappa2_p1_circuits.py')],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        self.assertIn('Ran 11 tests', out.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
