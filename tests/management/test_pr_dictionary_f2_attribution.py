"""Artificial saved-record tests only: no quantum/scientific packages."""
from fractions import Fraction
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks.pr_dictionary_validation import f2_attribution as analysis


def feature(kind, support=(), basis=None, frame=None, origins=('F0',)):
    return dict(kind=kind, support=list(support), basis_key=basis, frame_id=frame, origins=list(origins))


class SavedAttributionTests(unittest.TestCase):
    def setUp(self):
        self.atoms = [feature('IDENTITY'),
            feature('FRAME_Z', (0,), 'frame:a', 'F0'),
            feature('FRAME_Z', (1, 2), 'frame:b', 'F1', ('F1',)),
            feature('PHYSICAL_PAULI', (0,), 'local:ZZZZ', origins=('F0', 'F1')),
            feature('PHYSICAL_PAULI', (0, 1), 'local:XXZZ', origins=('F1',))]

    def tearDown(self):
        for name in ('qiskit', 'numpy', 'scipy', 'mpmath', 'trotterlib'):
            self.assertNotIn(name, sys.modules)

    def rows(self, n):
        for arm in analysis.ARMS:
            for i in range(n):
                for j in range(n):
                    for axis in ('X', 'Y'):
                        yield dict(id=analysis.case_id(arm, i, j, axis), arm=arm, axis=axis, i=i, j=j,
                                   metrics={m: 10+i+2*j for m in analysis.METRICS})
        for axis in ('X', 'Y'):
            yield dict(id=analysis.case_id('diagnostic', 0, 0, axis), arm='diagnostic', axis=axis, i=0, j=0,
                       metrics={m: 999 for m in analysis.METRICS})

    def test_partitions_conserve_ordered_case_counts_mass_and_saved_full_wrapper_cost(self):
        weights = {a: [Fraction(1, 5)]*5 for a in analysis.ARMS}
        result = analysis.analyze(self.rows(5), weights, {a: self.atoms for a in analysis.ARMS})
        self.assertEqual(result['physical_records'], 200)
        self.assertEqual(result['diagnostic_records'], 2)
        primary = {r['group']: r for r in result['groups'] if (r['view'], r['arm'], r['axis']) == ('mechanism', 'SS', 'X')}
        expected = dict(II=1, IF=4, IP=4, FF_same=2, FF_different=2, FP=8, PP=4)
        self.assertEqual({k: v['cases'] for k, v in primary.items()}, expected)
        self.assertEqual(sum(r['probability'] for r in primary.values()), 1)
        for r in primary.values():
            for m in analysis.METRICS:
                self.assertEqual(r['conditional'][m]*r['probability'], r['contribution'][m])
        self.assertEqual(result['totals']['SS']['X']['rz_count'], 16)
        delta, symmetric = analysis.differences(result['groups'], result['totals'])
        self.assertTrue(all(d['gap']['rz_count'] == 0 for d in delta))
        self.assertTrue(all(d['probability_mix']['rz_count']+d['conditional_cost']['rz_count'] == 0 for d in symmetric))

    def test_basis_recipe_and_joint_fragment_origin_are_not_confused(self):
        shared = analysis.classify(self.atoms[1], self.atoms[1])
        separate = analysis.classify(self.atoms[1], self.atoms[2])
        singleton = analysis.classify(self.atoms[0], self.atoms[1])
        self.assertEqual(shared['basis_relation'], 'frame_shared_full')
        self.assertEqual(separate['mechanism'], 'FF_different')
        self.assertEqual(singleton['frame_recipe'], 'F0_singleton_s1')
        paulis = analysis.classify(self.atoms[3], self.atoms[3])
        self.assertEqual(paulis['basis_relation'], 'pauli_shared_Z')
        self.assertEqual(paulis['origin_pair'], 'F0+F1>F0+F1')
        self.assertEqual(analysis.classify(self.atoms[3], self.atoms[4])['pauli_basis'], 'has_XY')

    def test_zero_probability_group_has_undefined_conditional_and_zero_contribution(self):
        weights = {a: [Fraction(1)] for a in analysis.ARMS}
        result = analysis.analyze(self.rows(1), weights, {a: [self.atoms[0]] for a in analysis.ARMS})
        empty = next(r for r in result['groups'] if (r['view'], r['arm'], r['axis'], r['group']) == ('mechanism', 'SS', 'X', 'FP'))
        self.assertEqual(empty['probability'], 0)
        self.assertIsNone(empty['conditional']['rz_count'])
        self.assertEqual(empty['contribution']['rz_count'], 0)

    def test_nonzero_symmetric_probability_and_conditional_terms_reconstruct_gap(self):
        weights = {a: [Fraction(3, 4), Fraction(1, 4)] for a in analysis.ARMS}
        weights['PS'] = [Fraction(1, 2), Fraction(1, 2)]
        rows = list(self.rows(2))
        for row in rows:
            if row['arm'] == 'diagnostic': continue
            has_frame = bool(row['i'] or row['j'])
            value = (40 if has_frame else 12) if row['arm'] == 'PS' else (30 if has_frame else 10)
            row['metrics'] = {m: value for m in analysis.METRICS}
        result = analysis.analyze(rows, weights, {a: self.atoms[:2] for a in analysis.ARMS})
        _, symmetric = analysis.differences(result['groups'], result['totals'])
        for row in symmetric:
            self.assertEqual(row['probability_mix']['rz_count'], Fraction(15, 2))
            self.assertEqual(row['conditional_cost']['rz_count'], Fraction(27, 4))

    def test_duplicate_missing_case_or_unnormalized_law_is_rejected(self):
        weights = {a: [Fraction(1)] for a in analysis.ARMS}; features = {a: [self.atoms[0]] for a in analysis.ARMS}
        rows = list(self.rows(1))
        with self.assertRaises(ValueError): analysis.analyze(iter(rows+[rows[0]]), weights, features)
        with self.assertRaises(ValueError): analysis.analyze(iter(rows[1:]), weights, features)
        with self.assertRaises(ValueError): analysis.analyze(iter(rows[:-1]), weights, features)
        with self.assertRaises(ValueError): analysis.analyze(iter(rows), {a: [Fraction(1, 2)] for a in analysis.ARMS}, features)


if __name__ == '__main__':
    unittest.main()
