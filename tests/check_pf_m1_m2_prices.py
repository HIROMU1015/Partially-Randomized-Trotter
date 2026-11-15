"""Fixed m1/m2 metadata and artificial equal-width/different-T/q gates."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'tests')]
import run_pf_m0_prices as M


class Metadata(unittest.TestCase):
    def test_exact_24_and_separate_rounds(self):
        plans, rows = M.selection((1, 2))
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({(r['configuration_id'],r['physical_id'],r['m'],r['axis']) for r in rows}), 24)
        self.assertEqual({m:sum(r['m']==m for r in rows) for m in (1,2)}, {1:12,2:12})
        self.assertTrue(all(len(p['cells'])==2 for ps in plans.values() for p in ps))
        for route in M.ROUTES:
            self.assertNotEqual(M.paths(route), M.paths(route,(1,2)))
        s2=[r for r in rows if r['formula']=='2nd' and r['route']=='FULL_PF' and r['axis']=='X']
        self.assertEqual([r['physical_candidate']['delta'] for r in s2],[.2,.2])
        self.assertEqual([r['physical_candidate']['q'] for r in s2],[1,2])
        self.assertEqual([r['physical_candidate']['T'] for r in s2],[.2,.4])
        self.assertEqual(len({r['physical_id'] for r in s2}),2)
        for bad in ((1,), (2,), (0,1,2), (2,1), (1,2,3)):
            with self.assertRaises(ValueError): M.scope(bad)

    def test_alignment_mutations_rejected(self):
        original=M.read
        for field in ('q','T','delta','id','axis','structure'):
            def changed(ref):
                d=deepcopy(original(ref))
                if ref==M.ALIGN:
                    row=d['rows'][8]  # Full S2 m2 X
                    if field in ('q','T','delta','id'): row['physical_candidate'][field]='wrong'
                    else: row[field]='wrong'
                return d
            M.read=changed
            try:
                with self.assertRaises((ValueError, TypeError, KeyError)): M.selection((1,2))
            finally:M.read=original


class Runtime(unittest.TestCase):
    def test_equal_width_keeps_distinct_time_scalar_and_wrapper(self):
        import h46_long_time_signal as signal
        signal.install()
        import h46_long_time_native as native
        import check_truncated_p7_prices_circuits as TT
        with TemporaryDirectory(prefix='pf_m1_m2_toy_') as directory:
            path=Path(directory);old=(native.PRIVATE,native.PUBLIC,native.compile_cached)
            ctx=TT.toy();native.PRIVATE=path;native.PUBLIC=path/'public';native.CONTEXT['TOY']=ctx
            def fake(wrapper,fingerprint,representative):
                return dict(metrics={k:len(wrapper.data) for k in native.METRICS}),wrapper.copy(),False
            native.compile_cached=fake
            try:
                M.S._RUN.update(native=native,ctx=ctx,trial=M.T.checked_trial,cohort='TOY_M1_M2',compile=fake,kw={})
                cells=[dict(m=m,row=TT.row('B0I',2,'2nd',T=T,q=q)) for m,T,q in ((1,.2,1),(2,.4,2))]
                results,res=M.processes(cells,path,2)
                self.assertEqual(len(results),2)
                self.assertTrue(all(r['ok'] and r['worker_exit_code']==0 and r['worker_signal'] is None for r in results))
                self.assertTrue(all(len(r['trial']['checks'])==7 for r in results))
                for axis in ('cosine','sine'):
                    self.assertEqual(len({r['wrappers'][axis]['fingerprint'] for r in results}),2)
                self.assertEqual({r['trial']['scalar'] for r in results},{ctx.ham.constant+ctx.preparations[2].extracted_identity_coefficient})
                self.assertEqual({r['trial']['technical_acceleration'][0]['builder_calls'] for r in results},{1,2})
                for p in path.glob('progress_*.jsonl'):
                    e=[json.loads(s) for s in p.read_text().splitlines()]
                    self.assertTrue(all(sum(x['stage']==s and x['state']=='START' for x in e)==sum(x['stage']==s and x['state']=='END' for x in e) for s in {x['stage'] for x in e}))
                self.assertEqual(res['not_started_after_failure'],0)
            finally:native.PRIVATE,native.PUBLIC,native.compile_cached=old


if __name__=='__main__':
    runtime='--runtime' in sys.argv
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Runtime if runtime else Metadata)
    ok=unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    if not runtime and ok:ok=subprocess.call([sys.executable,'-B',__file__,'--runtime'])==0
    sys.exit(0 if ok else 1)
