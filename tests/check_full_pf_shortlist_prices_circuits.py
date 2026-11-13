"""Artificial Full PF circuits through the frozen T3 checks + Full PF facts (provider-root process)."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import h46_long_time_signal as signal  # noqa: E402
signal.install()
import run_full_pf_shortlist_prices as FP  # noqa: E402
import check_truncated_p7_prices_circuits as TT  # noqa: E402  (same toy DF Hamiltonian/oracle)


class Tests(unittest.TestCase):
    def setUp(self):
        import h46_long_time_native as native
        self.native = native
        self.tmp = Path(tempfile.mkdtemp(prefix='fullpf_artificial_')); self.addCleanup(shutil.rmtree, self.tmp, True)
        native.PUBLIC, native.PRIVATE = self.tmp / 'public', self.tmp / 'private'; (self.tmp / 'private').mkdir()
        self.ctx = TT.toy(); native.CONTEXT['TOY'] = self.ctx; self.calls = []
        def fake(wrapper, fingerprint, representative):
            self.calls.append(fingerprint)
            return dict(metrics={k: 1 for k in native.METRICS}), None, False
        native.compile_cached = fake

    def test_full_rows_all_blocks_constant_scalar_and_signed_pieces(self):
        for formula, pieces in (('2nd', 1), ('4th', 3), ('4th(m5_best)', 11)):
            trial, facts, ok = FP.full_trial(self.native, self.ctx, TT.row('FULL', 3, formula), 'TOY')
            self.assertTrue(ok, facts)
            self.assertEqual(facts['retained_blocks'], 4)
            self.assertEqual(facts['tail_identity'], 0.)
            self.assertEqual(trial['scalar'], self.ctx.ham.constant)
            self.assertEqual(facts['signed_pieces'], 2 * pieces)
        self.assertEqual(len(self.calls), 6)

    def test_truncated_row_or_tail_identity_rejected_as_full(self):
        with self.assertRaises(ValueError):
            FP.full_trial(self.native, self.ctx, TT.row('FULL', 2, '2nd'), 'TOY')
        with self.assertRaises(ValueError):
            FP.full_trial(self.native, self.ctx, TT.row('B0I', 3, '2nd'), 'TOY')
        self.ctx.preparations[3] = self.ctx.preparations[2]
        with self.assertRaises(ValueError):
            FP.full_trial(self.native, self.ctx, TT.row('FULL', 3, '2nd'), 'TOY')


if __name__ == '__main__':
    unittest.main(verbosity=2)
