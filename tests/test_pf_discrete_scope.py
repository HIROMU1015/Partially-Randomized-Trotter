"""Exact PF grids use canonical metadata only; no provider or numerical body."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import pf_candidate_space as P
from trottertracks import pf_optimizer_connection as PF
from trottertracks import rpe_method_optimizer as O


class Reader:
    root = ROOT

    def __init__(self, index):
        self.index = index

    def document(self, commit, path):
        return deepcopy(self.index)

    def read(self, source):
        return deepcopy(self.index)


def problem():
    return dict(problem=dict(hamiltonian_data=dict(expected=dict(fragment_count=10))),
        evidence=dict(pf_candidate_index=dict(commit='ARTIFICIAL_INDEX', path='INDEX')),
        methods={name: dict(family='PF', index_method=index,
            formulas=['2nd', '4th', '4th(m5_best)'], delta_over_h_range=['2', '8'],
            delta_over_h_values=['2', '4', '8', '16'], caps=[1] * 13, p_values=prefixes)
            for name, index, prefixes in [('FULL_PF', 'FULL_PF', [10]),
                                          ('TRUNCATED_PF', 'B0I_P7', [6, 7, 8, 9])]})


class DiscretePF(unittest.TestCase):
    def test_exact_cartesian_grid_includes_unindexed_sixteen_for_all_formulas(self):
        # The saved index deliberately omits sixteen, as it does for S2/Yoshida4.
        scope = problem()
        reader = Reader(dict(candidates=[dict(method=m, formula=f, requested_delta_over_h=w)
            for m in ('FULL_PF', 'B0I_P7') for f in scope['methods']['FULL_PF']['formulas']
            for w in ('2', '4', '8')]))
        rows = []
        with patch.object(P.C, 'execute', side_effect=AssertionError('Scientific provider called')):
            PF.enumerate_prefixes(ROOT, reader, scope, rows)
        self.assertEqual(len(rows), 60)
        self.assertEqual(len({r['configuration_id'] for r in rows}), 60)
        self.assertEqual(sum(r['method'] == 'FULL_PF' for r in rows), 12)
        self.assertEqual(sum(r['method'] == 'TRUNCATED_PF' for r in rows), 48)
        expected = {(p, f, w) for p in (6, 7, 8, 9, 10)
                    for f in scope['methods']['FULL_PF']['formulas'] for w in (2, 4, 8, 16)}
        self.assertEqual({(r['key']['p'], r['key']['formula'], Fraction(r['key']['delta_over_h']))
                          for r in rows}, expected)
        self.assertTrue(all(not r['ready'] and r['prices'] is None for r in rows))
        for row in rows:
            cfg = row['configuration_body']
            self.assertEqual(cfg['id'], row['configuration_id'])
            self.assertEqual(P.exact(cfg['spec']['delta_pf']),
                             P.H * Fraction(row['key']['delta_over_h']))
            # Large requested steps retain the mapper's shorter early rounds.
            if row['key']['delta_over_h'] == '16':
                self.assertEqual(cfg['rounds'][0]['q_PF'], 1)
                self.assertEqual(P.exact(cfg['rounds'][0]['actual_delta_pf']), P.H)

    def test_saved_index_catalog_and_registry_exclude_intermediate_widths(self):
        scope = problem()
        scope['methods'] = {'TRUNCATED_PF': scope['methods']['TRUNCATED_PF']}
        index = dict(candidates=[dict(method='B0I_P7', formula='2nd', requested_delta_over_h=w,
            configuration_id='indexed-' + w, source_bound_13round_packet=None)
            for w in ('2', '3', '4', '16/3', '16')])
        catalog = dict(candidates=[dict(method='TRUNCATED_PF', L_D=p, formula='2nd',
            delta_pf=str(Fraction(w) / 10), h='1/10', configuration_id='saved-' + str(p) + '-' + w,
            label='synthetic', point_id='synthetic-point')
            for p in (5, 6, 9) for w in ('4', '5', '16')])
        with patch.object(O, 'configuration_prices', return_value=(None, 'ARTIFICIAL_NO_PRICE')):
            rows = O.enumerate_candidates(Reader(index), scope, dict(fragment_count=10), catalog)
        self.assertEqual({r['configuration_id'] for r in rows},
                         {'indexed-2', 'indexed-4', 'indexed-16',
                          'saved-6-4', 'saved-6-16', 'saved-9-4', 'saved-9-16'})
        self.assertFalse(PF._inside(dict(formula='2nd', p=7, delta_over_h='3'), scope['methods']['TRUNCATED_PF']))
        self.assertTrue(PF._inside(dict(formula='2nd', p=7, delta_over_h='16'), scope['methods']['TRUNCATED_PF']))
        self.assertFalse(PF._inside(dict(formula='2nd', p=5, delta_over_h='4'), scope['methods']['TRUNCATED_PF']))

    def test_existing_allowlist_and_formula_restrictions_are_preserved(self):
        scope = problem()
        scope['methods']['FULL_PF'].update(formulas=['2nd'], configuration_ids=[])
        rows = []
        PF.enumerate_prefixes(ROOT, Reader(dict(candidates=[])), scope, rows, methods=['FULL_PF'])
        self.assertEqual(rows, [])

    def test_legacy_intervals_and_exact_validation(self):
        self.assertTrue(PF.width_inside('3', dict(delta_over_h_range=['2', '4'])))
        self.assertFalse(PF.width_inside('5', dict(delta_over_h_range=['2', '4'])))
        self.assertEqual(PF.width_values(dict(delta_over_h_values=['2', '2/1', '16'])),
                         (Fraction(2), Fraction(16)))
        for values in ([], '2', ['0'], ['-1'], [True]):
            with self.assertRaises(ValueError):
                PF.width_values(dict(delta_over_h_values=values))


if __name__ == '__main__':
    unittest.main()
