"""Stage child runner on artificial circuits (provider-root process).

Toy DF Hamiltonian and the unchanged T3/Full PF trial functions with a counting
compile_cached in forked workers: per-cell results, both wrapper fingerprints,
qubit counts, failed-cell partial save and no start after a failure. No H4 input,
transpile or price.
"""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import h46_long_time_signal as signal  # noqa: E402
signal.install()
import pf_price_stages as S  # noqa: E402
import run_full_pf_shortlist_prices as FP  # noqa: E402
import run_truncated_p7_prices as T3  # noqa: E402
import check_truncated_p7_prices_circuits as TT  # noqa: E402  (toy DF Hamiltonian/oracle)


class Tests(unittest.TestCase):
    def setUp(self):
        import h46_long_time_native as native
        self.native = native
        self.tmp = Path(tempfile.mkdtemp(prefix='pf_stage_child_')); self.addCleanup(shutil.rmtree, self.tmp, True)
        native.PUBLIC, native.PRIVATE = self.tmp / 'public', self.tmp / 'private'; (self.tmp / 'private').mkdir()
        self.ctx = TT.toy(); native.CONTEXT['TOY'] = self.ctx

        def fake(wrapper, fingerprint, representative):
            return dict(metrics={k: len(wrapper.data) for k in native.METRICS}), None, False
        self.original = native.compile_cached
        native.compile_cached = fake
        self.addCleanup(setattr, native, 'compile_cached', self.original)

    def test_parallel_cells_keep_trial_and_record_both_wrappers(self):
        cells = [dict(row=TT.row('B0I', 2, f, q=q)) for f in ('2nd', '4th', '4th(m5_best)') for q in (1, 2)]
        results, res = S.run_cells(self.native, self.ctx, T3.checked_trial, 'TOY_STAGE', cells, workers=3)
        self.assertEqual((len(results), res['started'], res['not_started_after_failure']), (6, 6, 0))
        self.assertEqual([r['cell']['row']['id'] for r in results], [c['row']['id'] for c in cells])
        for r in results:
            self.assertTrue(r['ok'], r.get('error'))
            w = r['wrappers']
            self.assertEqual(set(w), {'cosine', 'sine'})
            self.assertNotEqual(w['cosine']['fingerprint'], w['sine']['fingerprint'])
            self.assertEqual(w['cosine']['wrapper_qubits'], w['sine']['wrapper_qubits'])
            self.assertGreater(w['cosine']['wrapper_qubits'], 4)
            self.assertEqual(r['trial']['scalar'], self.ctx.ham.constant + self.ctx.preparations[2].extracted_identity_coefficient)
        self.assertEqual(self.native.compile_cached.__name__, 'fake')  # parent module untouched

    def test_full_pf_trial_in_workers(self):
        cells = [dict(row=TT.row('FULL', 3, f)) for f in ('2nd', '4th')]
        results, _ = S.run_cells(self.native, self.ctx, FP.full_trial, 'TOY_STAGE_FULL', cells, workers=2)
        self.assertTrue(all(r['ok'] for r in results))
        self.assertEqual({r['trial']['scalar'] for r in results}, {self.ctx.ham.constant})

    def test_failed_cell_saved_and_no_new_start_after_failure(self):
        bad = dict(row=TT.row('FULL', 2, '2nd'))  # not all blocks: full_trial refuses before compiling
        cells = [bad] + [dict(row=TT.row('FULL', 3, '4th(m5_best)', q=q)) for q in range(1, 9)]
        results, res = S.run_cells(self.native, self.ctx, FP.full_trial, 'TOY_STAGE_STOP', cells, workers=1)
        self.assertFalse(results[0]['ok'])
        self.assertIn('ValueError', results[0]['error'])
        self.assertLess(res['started'], len(cells))
        self.assertEqual(res['not_started_after_failure'], len(cells) - len(results))
        ids = [r['cell']['row']['id'] for r in results]
        self.assertEqual(len(ids), len(set(ids)))  # no retry


if __name__ == '__main__':
    unittest.main(verbosity=2)
