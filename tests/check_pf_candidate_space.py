"""Metadata-only PF prefix/width tests; no real preparation, operators or Q3."""
import csv
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import pf_candidate_space as P


class CandidateSpace(unittest.TestCase):
    def test_full_twenty_canonical_ids_preserved(self):
        old = json.loads(P.I.C.blob(ROOT, '438d6c9a4',
            'docs/research/h4_pf_optimizer_search_20261116/CANDIDATES_AT_START.json'))
        expected = {r['configuration_id'] for r in old if r['method'] == 'FULL_PF'}
        actual = P.candidates(ROOT, p_values=[10])
        self.assertEqual(len(actual), 20)
        self.assertEqual({r['configuration_id'] for r in actual}, expected)

    def test_saved_truncated_ids_preserved(self):
        raw = P.I.C.blob(ROOT, P.T2_SOURCE, P.T2_DIR+'CONFIGURATIONS.csv').decode()
        rows = tuple(csv.DictReader(io.StringIO(raw)))
        for row in rows:
            c = P.configure(ROOT, 'TRUNCATED_PF', int(row['prefix']), row['formula'], row['requested_delta_over_h'])
            self.assertEqual(c['id'], row['configuration_id'])

    def test_p_zero_and_ten_alias_and_independent_prefixes(self):
        configs = [P.configure(ROOT, 'B0I', p, '2nd', '1') for p in range(11)]
        self.assertEqual(len({c['id'] for c in configs}), 11)
        for p, c in enumerate(configs[:10]):
            self.assertEqual(c['spec']['L_D'], p)
            self.assertEqual(c['spec']['split']['retained_hamiltonian']['prefix'], p)
            self.assertEqual(c['spec']['split']['original_PF_method'], 'B0I')
        self.assertEqual(configs[0]['rounds'][0]['candidate']['prefix'], 0)
        self.assertEqual(configs[10], P.configure(ROOT, 'FULL_PF', 10, '2nd', '1'))

    def test_arbitrary_width_signed_structure_and_profile(self):
        c = P.configure(ROOT, 'B0I', 8, '4th', '11/3')
        self.assertEqual(P.exact(c['spec']['delta_pf']), P.H*Fraction(11, 3))
        self.assertTrue(any(w < 0 for w in c['rounds'][0]['structure']['signed_weights']))
        self.assertEqual(c['spec']['profile']['P'], 0)
        self.assertEqual(c['spec']['profile']['qiskit_version'], '1.3.0')
        self.assertEqual(c['spec']['profile']['control'], 'symmetric_directional')
        self.assertEqual(c['spec']['profile']['wrapper'], 'full_Hadamard_with_measurement')
        self.assertEqual(c['spec']['profile']['basis_gates'], ['rz', 'sx', 'x', 'cx'])
        self.assertEqual(c['spec']['profile']['optimization_level'], 1)
        self.assertEqual(c['spec']['profile']['seed_transpiler'], 17)

    def test_saved_priority_values_only_and_all_42_cell_locators(self):
        seen = set()
        for p in range(10):
            rows = P.candidates(ROOT, p_values=[p])
            t1 = rows[0]['priority_evidence']['T1']
            saved = json.loads(P.I.C.blob(ROOT, t1['source']['commit'], t1['source']['path']))
            self.assertEqual(t1['delta_E_trunc_B0I_ha'], saved['delta_E_trunc_B0I_ha'])
            self.assertEqual(t1['tail_identity_scalar'], saved['tail_coefficients']['used_c_tail'])
            self.assertEqual(t1['lambda_R'], saved['tail_coefficients']['used_lambda_R'])
            for row in rows:
                self.assertEqual(row['priority_evidence']['usage'], 'PRIORITY_ONLY_NOT_Q3_ACCEPTANCE_OR_REJECTION')
                for cell in row['priority_evidence']['T2']:
                    src = cell['scope_source'];seen.add((src['commit'], src['path']))
        self.assertEqual(len(seen), 42)
        for commit, path in seen:
            scope = json.loads(P.I.C.blob(ROOT, commit, path))
            self.assertIn(scope['prefix'], (7, 8, 9))

    def test_no_numeric_entrypoint_or_array_read(self):
        original_blob = P.I.C.blob
        def metadata_only(root, commit, path):
            if path.endswith(('.npz', '.qpy')):
                raise AssertionError('Candidate metadata attempted to read scientific arrays/circuits')
            return original_blob(root, commit, path)
        with patch.object(P.I.C, 'blob', metadata_only), patch.object(P.I, 'produce_signals', side_effect=AssertionError('science')):
            rows = P.candidates(ROOT, formulas=['2nd'], widths=['1', '1/1', '3/2'], p_values=[0, 7, 10])
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({r['configuration_id'] for r in rows}), 6)

    def test_invalid_scope_rejected(self):
        for method, p, formula, width in (('FULL_PF', 7, '2nd', '1'),
                ('B0', 7, '2nd', '1'), ('B0I', -1, '2nd', '1'),
                ('B0I', True, '2nd', '1'), ('B0I', 11, '2nd', '1'),
                ('B0I', 7, '8th', '1'), ('B0I', 7, '2nd', '0'),
                ('B0I', 7, '2nd', True)):
            with self.assertRaises(ValueError):
                P.configure(ROOT, method, p, formula, width)
        for values in ([7, 7], [-1], [True]):
            with self.assertRaises(ValueError): P.candidates(ROOT, p_values=values)


if __name__ == '__main__':
    unittest.main(verbosity=2)
