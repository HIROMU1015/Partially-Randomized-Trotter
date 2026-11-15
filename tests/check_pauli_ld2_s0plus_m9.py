#!/usr/bin/env python3
"""Synthetic tests for scripts/pauli_ld2_s0plus_m9.py (S0+ additions only).

Uses the toy fixture of tests/check_pauli_ld2_step_audit.py (no transpile, no saved input):
the S0-r1 audit is run on toy paths, a frozen reference is built from two toy paths, and the
reference check / decomposition / statistics are applied to another toy path.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import check_pauli_ld2_step_audit as T  # noqa: E402
import pauli_ld2_s0plus_m9 as S  # noqa: E402
import pauli_ld2_step_audit as A  # noqa: E402

from qiskit import QuantumCircuit  # noqa: E402


def toy_run(mutate=None):
    tmp = tempfile.TemporaryDirectory()
    fx = T.Fixture(tmp.name, mutate=mutate)
    fx.run()
    edges = json.loads((A.RUN / 'EDGES.json').read_text())['per_wire']
    n = len(edges)
    arrays = {p['path_id']: A.load_x_arrays(p['path_id'], p['q'], n) for p in fx.paths}
    return tmp, fx, edges, arrays


class S0PlusTests(unittest.TestCase):
    def test_01_reference_matches_and_decomposition_restores_total(self):
        tmp, fx, edges, arrays = toy_run()
        with tmp:
            ref = S.make_reference(edges, arrays['S66/p0'], arrays['Q8R66/p0'])
            chk = S.check_against_reference(arrays['M5_ROUND/p0'], 2, ref)
            self.assertEqual(chk['status'], 'PASS', chk)
            const = {k: sum(w[k + '_rz'] for w in ref) for k in ('F', 'P', 'S', 'E')}
            intercept = const['F'] + const['S'] + const['E'] + 2 - (const['P'] + const['S'])
            per_step = const['P'] + const['S']
            saved = next(p for p in fx.paths if p['path_id'] == 'M5_ROUND/p0')['axes']['X']['metrics']['rz_count']
            restored = intercept + per_step * 2 + chk['first_body_rz'] + sum(chk['later_body_rz'])
            self.assertEqual(restored, saved)

    def test_02_suffix_mismatch_is_located(self):
        tmp, fx, edges, arrays = toy_run()
        with tmp:
            ref = S.make_reference(edges, arrays['S66/p0'], arrays['Q8R66/p0'])
            bad = [dict(w) for w in ref]
            bad[1] = dict(bad[1], S=list(bad[1]['S']))
            bad[1]['S'][-1] = '0' * 16          # corrupt the last suffix token of wire 1
            chk = S.check_against_reference(arrays['Q8R66/p0'], 3, bad)
            self.assertEqual(chk['status'], 'FAIL')
            self.assertEqual(chk['mismatch']['step'], 1)
            self.assertEqual(chk['mismatch']['wire'], 1)
            self.assertEqual(chk['mismatch']['part'], 'suffix')
            self.assertEqual(chk['mismatch']['position_from_end'], 1)
            # statistics stay defined from segment RZ minus the reference constants
            self.assertEqual(len(chk['later_body_rz']), 2)

    def test_03_entry_mismatch_is_located(self):
        tmp, fx, edges, arrays = toy_run()
        with tmp:
            ref = S.make_reference(edges, arrays['S66/p0'], arrays['Q8R66/p0'])
            bad = [dict(w) for w in ref]
            bad[0] = dict(bad[0], P=list(bad[0]['P']))
            if not bad[0]['P']:
                self.skipTest('toy later prefix empty on wire 0')
            bad[0]['P'][0] = '0' * 16
            chk = S.check_against_reference(arrays['Q8R66/p0'], 3, bad)
            self.assertEqual(chk['mismatch'], dict(step=2, wire=0, part='entry', position=0))

    def test_04_describe(self):
        st = S.describe([10, 12, 14], 2)
        self.assertEqual((st['n'], st['mean'], st['sample_variance'], st['mean_over_R']), (3, 12.0, 4.0, 6.0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
