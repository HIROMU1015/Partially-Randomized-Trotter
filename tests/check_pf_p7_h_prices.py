"""Fixed candidate binding and synthetic guard; actual worker tested on toy DF."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import run_pf_p7_h_prices as P


class Metadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan,cls.cells,cls.rows=P.template()

    def test_exact_candidate_and_four_axes(self):
        self.assertEqual(self.plan['configuration']['id'],P.CONFIG)
        self.assertEqual({(r['m'],r['axis']) for r in self.rows},{(1,'X'),(1,'Y'),(2,'X'),(2,'Y')})
        self.assertEqual([(c['row']['T'],c['row']['q'],c['row']['delta']) for c in self.cells],[(.2,2,.1),(.4,4,.1)])
        job=dict(route='TRUNCATED_PF',configuration_id=P.CONFIG,cohort=P.COHORT,cells=self.cells,rounds=[1,2])
        P.validate_job(job)
        for name,value in [('configuration_id','wrong'),('cohort','old'),('route','FULL_PF')]:
            with self.assertRaises(ValueError):P.validate_job(dict(job,**{name:value}))
        for name,value in [('q',1),('delta',.2),('prefix',10),('id','old_alias')]:
            bad=deepcopy(job);bad['cells'][0]['row'][name]=value
            with self.assertRaises((ValueError,KeyError)):P.validate_job(bad)
        bad=deepcopy(job);bad['cells']*=2
        with self.assertRaises(ValueError):P.validate_job(bad)
        bad=deepcopy(job);bad['cells'][0]['m']=2
        with self.assertRaises(ValueError):P.validate_job(bad)

    def test_original_six_axes_keep_price_producer_and_cohort(self):
        reused=P.reuse_trials()
        self.assertEqual(set(reused),{'0','3','4'})
        self.assertEqual([reused[m]['trial']['prices']['cosine']['rz_count'] for m in ('0','3','4')],[5820,44411,88515])
        self.assertEqual(reused['4']['cohort'],'EXPLORATION')
        self.assertIsNone(reused['3']['original_cohort'])
        self.assertNotEqual(reused['4']['cohort'],P.COHORT)

    def test_existing_record_builder_and_guard_on_synthetic_price(self):
        P.M.T._meta()
        from trottertracks import price_implementation_guard as G
        trial=dict(scalar=1.25,prices={a:dict(rz_count=11,rz_depth=7,cx_count=5,cx_depth=4,total_depth=19,circuit_size=31) for a in ('cosine','sine')})
        source=dict(commit=P.PLAN_COMMIT,role='SYNTHETIC_ONLY')
        prices,sides,proof=P.records(self.plan,self.cells[0],trial,source,'ARTIFICIAL_COHORT')
        cfg=self.plan['configuration']
        policy=dict(id='SYNTHETIC_ONLY',source=P.M.S.GUARD_SOURCE,profile=cfg['spec']['profile'],by_method=dict(TRUNCATED_PF=dict(native_policy=P.M.T.NATIVE_POLICY,rte_rotation_lowering=P.M.T.NO_RTE,scalar_quantity=P.M.T.SCALAR_QUANTITY)))
        cohorts={(cfg['rounds'][p['m']]['candidate']['id'],p['axis']):p['cohort'] for p in prices}
        def evidence(root,ref):return proof[ref['selector'][0]]
        with patch.object(G,'evidence',evidence):
            result=G.check(ROOT,cfg,prices,cohorts=cohorts,bindings=sides,policy=policy)
            self.assertEqual({(r['m'],r['axis']) for r in result['checked']},{(1,'X'),(1,'Y')})
            for field in ('axis','source','quantity','physical_context'):
                bad=deepcopy(prices);bindings=deepcopy(sides)
                if field=='axis':
                    bad[0][field]='wrong'
                    # The existing guard leaves missing axes to its caller.
                    missing=G.check(ROOT,cfg,bad,cohorts=cohorts,bindings=bindings,policy=policy)
                    self.assertNotEqual(len(missing['checked']),len(bad));continue
                elif field=='physical_context':bindings[0][field]['physical_candidate']['delta']=.2
                else:bad[0][field]='wrong'
                with self.assertRaises((ValueError,AttributeError)):
                    G.check(ROOT,cfg,bad,cohorts=cohorts,bindings=bindings,policy=policy)


if __name__=='__main__':
    ok=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Metadata)).wasSuccessful()
    # A separate provider-first process avoids metadata import resolution.
    if ok:ok=subprocess.call([sys.executable,'-B',str(ROOT/'tests/check_pf_m0_prices.py'),'--runtime'])==0
    sys.exit(0 if ok else 1)
