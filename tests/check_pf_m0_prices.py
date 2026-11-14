"""Fixed metadata and observed fork entry on artificial DF; no real H4 calls."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'tests')]
import run_pf_m0_prices as M


class Metadata(unittest.TestCase):
    def test_fixed_twelve_and_route_separation(self):
        chosen, rows = M.selection()
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({(r['configuration_id'], r['axis']) for r in rows}), 12)
        self.assertEqual({r['physical_candidate']['q'] for r in rows}, {1})
        self.assertEqual({r['physical_candidate']['delta'].hex() for r in rows}, {(.1).hex()})
        self.assertNotEqual(M.paths('FULL_PF'), M.paths('TRUNCATED_PF'))
        for route in M.ROUTES:
            self.assertEqual(len(chosen[route]), 3)
        self.assertEqual(M.S.STAGES, dict(CALIBRATION=(4, 5), HOLDOUT=(6,)))

    def test_mutated_physical_axis_and_signed_structure_rejected(self):
        original = M.read
        for mutation in ('q', 'axis', 'structure', 'configuration_id'):
            def changed(ref):
                doc = deepcopy(original(ref))
                if ref == M.ALIGN:
                    if mutation == 'q': doc['rows'][0]['physical_candidate']['q'] = 2
                    elif mutation == 'structure': doc['rows'][0]['structure'] = {}
                    else: doc['rows'][0][mutation] = 'wrong'
                return doc
            M.read = changed
            try:
                with self.assertRaises((ValueError, KeyError)): M.selection()
            finally: M.read = original


class Runtime(unittest.TestCase):
    def test_abnormal_exit_preserved_without_retry(self):
        import os
        with TemporaryDirectory(prefix='pf_m0_exit_') as directory:
            original = M.worker
            def failure(*args): os._exit(7)
            M.worker = failure
            try:
                results, res = M.processes([dict(row={'id':'SYNTHETIC'})]*2, Path(directory), 1)
                self.assertEqual((len(results), res['not_started_after_failure']), (1, 1))
                self.assertEqual(results[0]['worker_exit_code'], 7)
                self.assertFalse(results[0]['ok'])
            finally: M.worker = original

    def test_actual_worker_entry_fast_context_observation_and_exit(self):
        import h46_long_time_signal as signal
        signal.install()
        import h46_long_time_native as native
        import check_truncated_p7_prices_circuits as TT
        from trottertracks.resource_applicability import ax2b_native_df_v5 as api
        import h46_b3_fast_fingerprint as fast
        with TemporaryDirectory(prefix='pf_m0_toy_') as directory:
            path = Path(directory); old = (native.PRIVATE, native.PUBLIC, native.compile_cached)
            ctx = TT.toy(); native.PRIVATE = path; native.PUBLIC = path/'public'; native.CONTEXT['TOY'] = ctx
            def fake(wrapper, fingerprint, representative):
                # Only artificial circuits, no transpile or scientific H4 input.
                return dict(metrics={k: len(wrapper.data) for k in native.METRICS}), wrapper.copy(), False
            native.compile_cached = fake
            try:
                M.S._RUN.update(native=native, ctx=ctx, trial=M.T.checked_trial, cohort='TOY_M0', compile=fake, kw={})
                cells = [dict(row=TT.row('B0I', 2, f, T=.1, q=1)) for f in ('2nd', '4th', '4th(m5_best)')]
                results, res = M.processes(cells, path, 3)
                self.assertEqual((len(results), res['not_started_after_failure']), (3, 0))
                for result in results:
                    self.assertTrue(result['ok'], result)
                    self.assertEqual((result['worker_exit_code'], result['worker_signal']), (0, None))
                    self.assertEqual(len(result['trial']['checks']), 7)
                    self.assertNotEqual(result['wrappers']['cosine']['fingerprint'], result['wrappers']['sine']['fingerprint'])
                    self.assertEqual(result['trial']['technical_acceleration'][0]['builder_calls'], {'2nd':1, '4th':3, '4th(m5_best)':11}[result['cell']['row']['formula']])
                for p in path.glob('progress_*.jsonl'):
                    events = [json.loads(line) for line in p.read_text().splitlines()]
                    stages = {e['stage'] for e in events}
                    self.assertTrue({'cell_trial', 'compose', 'measured_wrapper', 'fingerprint', 'compile_and_archive', 'statevector_gate'} <= stages)
                    for stage in stages:
                        mine = [e for e in events if e['stage'] == stage]
                        self.assertEqual(sum(e['state']=='START' for e in mine), sum(e['state']=='END' for e in mine))
                    self.assertTrue(all(e['seconds'] >= 0 for e in events if e['state']=='END'))
                # Parent aliases are untouched by the forked instrumentation.
                self.assertIs(native.compile_cached, fake)
                self.assertIsNotNone(api.build_native_hadamard_wrapper)
                self.assertEqual(fast.FingerprintSession.fingerprint.__name__, 'fingerprint')
                with self.assertRaises(RuntimeError):
                    with M.span(path/'error.jsonl', 'synthetic_failure'): raise RuntimeError('synthetic')
                self.assertEqual(json.loads((path/'error.jsonl').read_text().splitlines()[-1])['state'], 'ERROR')
            finally:
                native.PRIVATE, native.PUBLIC, native.compile_cached = old


if __name__ == '__main__':
    # The production runner likewise execs a fresh provider-first process.
    import subprocess
    runtime = '--runtime' in sys.argv
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Runtime if runtime else Metadata)
    ok = unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    if not runtime and ok:
        ok = subprocess.call([sys.executable, '-B', __file__, '--runtime']) == 0
    sys.exit(0 if ok else 1)
