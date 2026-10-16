"""Artificial readiness guards; no molecular inputs or real cost fitting."""
from copy import deepcopy
from fractions import Fraction
import unittest

from h8_phase_b1_contract import (
    PROFILE, COMPILER, REQUIRED, b2_axis_passes, inspect_anchors,
    require_full_b1_provenance, select_b2, validate_draft,
)


def anchors():
    return [dict(method=m, prefix=p, q=q, axis=a, formula='2nd',
                 saved_RZ=100, control='symmetric_directional',
                 runtime_profile_sha256=PROFILE, compiler_hash=COMPILER,
                 evidence={k: True for k in REQUIRED})
            for m in ('B0', 'B0I') for p in (14, 21) for q in (1, 2, 4, 8)
            for a in ('cosine', 'sine')]


class ReadinessTests(unittest.TestCase):
    def test_complete_metadata_still_does_not_fit(self):
        r = require_full_b1_provenance(anchors())
        self.assertEqual(r['provenance_eligible_axes'], 32)
        self.assertFalse(r['actual_fit_performed'])
        self.assertFalse(r['actual_design_rank_evaluated'])

    def test_each_missing_evidence_blocks(self):
        for key in REQUIRED:
            with self.subTest(key=key):
                rows = anchors()
                same = [r for r in rows if r['method'] == 'B0' and r['axis'] == 'cosine']
                for r in same[3:]:
                    r['evidence'][key] = False
                rows[0]['evidence'][key] = False
                with self.assertRaisesRegex(ValueError, 'STOP_MISSING'):
                    require_full_b1_provenance(rows)

    def test_no_partial_family_fallback(self):
        rows = anchors()
        for row in rows:
            if row['method'] == 'B0':
                row['evidence']['runtime_profile_bound'] = False
        self.assertEqual(inspect_anchors(rows)['counts']['B0I/cosine'], 8)
        with self.assertRaisesRegex(ValueError, 'STOP_MISSING'):
            require_full_b1_provenance(rows)

    def test_old_profile_not_backfilled(self):
        for field in ('runtime_profile_sha256', 'compiler_hash'):
            rows = anchors()
            for row in rows:
                if row['method'] == 'B0': row[field] = 'old-or-unknown'
            with self.assertRaisesRegex(ValueError, 'STOP_MISSING'):
                require_full_b1_provenance(rows)

    def test_duplicate_missing_wrong_grid_rejected(self):
        for change in ('duplicate', 'missing', 'grid', 'formula', 'control', 'RZ'):
            rows = anchors()
            if change == 'duplicate': rows.append(deepcopy(rows[0]))
            elif change == 'missing': rows.pop()
            elif change == 'grid': rows[0]['prefix'] = 7
            elif change == 'formula': rows[0]['formula'] = '4th'
            elif change == 'control': rows[0]['control'] = 'ordinary'
            else: rows[0]['saved_RZ'] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                inspect_anchors(rows)

    def test_fixed_b2_selection_ignores_ineligible(self):
        cells = [dict(id=f'{m}_p{p}_q{q}_4th', method=m, prefix=p, q=q,
                      formula='4th', eligible=(p != 15 or q != 1))
                 for m in ('B0', 'B0I') for p in (15, 21) for q in (1, 8)]
        r = select_b2(cells)
        self.assertEqual([a['candidate_id'] for a in r],
                         ['B0_p15_q8_4th'] * 2 + ['B0I_p21_q8_4th'] * 2)
        self.assertTrue(all(a['predicted_RZ'] is None and not a['compile_authorized'] for a in r))
        self.assertEqual(select_b2([]), [])

    def test_b2_boundaries_independent_axes(self):
        self.assertTrue(b2_axis_passes(Fraction(10199, 100), 100))
        self.assertFalse(b2_axis_passes(102, 100))
        self.assertFalse(b2_axis_passes(98, 100))
        self.assertFalse(b2_axis_passes(103, 100))
        # Measured denominator: 2/102 < 2%, 2/100 == 2%.
        self.assertTrue(b2_axis_passes(100, 102))
        with self.assertRaises(ValueError): b2_axis_passes(100, 0)

    def test_published_draft_authorization_and_model_guards(self):
        from pathlib import Path
        import json
        root = Path(__file__).resolve().parents[1]
        p = root / 'artifacts/resource_applicability/track_a_h8_phase_b1_readiness/2026-10-16/execution_contract_DRAFT_v1.json'
        plan = json.loads(p.read_text())
        self.assertTrue(validate_draft(plan))
        for section, key, value in [('authorization', 'approved', True),
                ('authorization', 'SOURCE', 'invented'), ('authorization', 'retry', True),
                ('models', 'H6_coefficients_transferred', True), ('models', 'coefficients', [1, 2, 3]),
                ('B2', 'compile_concurrency', 2), ('B2', 'max_axes', 6),
                ('B2', 'refit', True), ('C', 'separate_approval', False),
                ('implementation', 'real_fit_runner_sealed', True)]:
            changed = deepcopy(plan)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_draft(changed)


if __name__ == '__main__':
    unittest.main()
