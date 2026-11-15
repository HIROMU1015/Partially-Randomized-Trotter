"""Coarse-width metadata, preserved receipts and artificial calibration only."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import run_p7_y4_coarse_prices as A
import check_pf_p7_h_prices as Previous


class CoarsePrices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan,cls.cells,cls.rows=A.template()

    def test_fixed_six_axes_two_roles_and_legacy_scope(self):
        self.assertEqual([(m,c['row']['q'],c['row']['delta'].hex()) for m,c in self.cells.items()],[(5,5,(.64).hex()),(6,10,(.64).hex()),(9,80,(.64).hex())])
        self.assertEqual(len(self.rows),26)
        for stage,rounds in A.STAGES.items():
            job=dict(stage=stage,route='TRUNCATED_PF',configuration_id=A.CONFIG,rounds=list(rounds),cells=[self.cells[m] for m in rounds],cohort='D_P7_Y4_32O5_'+stage+'_V1')
            A.validate_job(job)
            for field,value in [('configuration_id','old'),('rounds',[0]),('cohort','mixed'),('route','FULL_PF')]:
                with self.assertRaises(ValueError):A.validate_job(dict(job,**{field:value}))
            wrong=deepcopy(job);wrong['cells'][0]['row']['q']+=1
            with self.assertRaises(ValueError):A.validate_job(wrong)
        self.assertEqual(A.M.scope((0,))[2],'M0_DIRECT_ONLY')
        with self.assertRaises(ValueError):A.M.scope((5,6))

    def test_saved_ten_axes_and_native_SOURCE_exception_are_explicit(self):
        values=A.saved_early(self.plan)
        self.assertEqual({(r['m'],r['axis']) for r in values},{(m,a) for m in range(5) for a in ('X','Y')})
        self.assertTrue(all(r['fit_role']=='EXCLUDED_EARLY_WIDTH_DIFFERS' for r in values))
        self.assertTrue(all(r['physical_candidate']['delta']!=.64 for r in values))
        self.assertEqual([r['native_reuse_basis'] for r in values if r['m']==3],['EXISTING_C_ADMISSION_EQUIVALENCE']*2)
        real=A.read
        def wrong(ref):
            value=deepcopy(real(ref))
            if ref.get('path','').endswith('/m0_direct/run01/PRICES.json') and ref.get('selector')==[2]:
                value['profile_extension']['source']['commit']='wrong'
            return value
        with patch.object(A,'read',side_effect=wrong):
            with self.assertRaises(ValueError):A.saved_early(self.plan)

    def test_real_child_entry_passes_only_verified_phase_to_shared_native_child(self):
        job=dict(stage='HOLDOUT',route='TRUNCATED_PF',configuration_id=A.CONFIG,rounds=[9],cells=[self.cells[9]],cohort='D_P7_Y4_32O5_HOLDOUT_V1',execution_SOURCE='TOY')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'job.json';path.write_text(json.dumps(job))
            with patch.object(A.subprocess,'check_output',side_effect=lambda cmd,**kw:(ROOT/cmd[2].split(':',1)[1]).read_bytes()),patch.object(A.M,'child',return_value=0) as native:
                self.assertEqual(A.child(path),0)
                verify=native.call_args.kwargs['validate_rounds'];verify([9])
                with self.assertRaises(ValueError):verify([5,6])

    def test_axis_separate_exact_two_point_fit_and_early_holdout_rejection(self):
        prices=[]
        for m in (5,6):
            q=self.cells[m]['row']['q'];trial=dict(scalar=1.25,prices={k:dict(rz_count=(7+3*q if k=='cosine' else 11+5*q),rz_depth=1,cx_count=1,cx_depth=1,total_depth=1,circuit_size=1) for k in ('cosine','sine')})
            records,_,_=A.P.records(self.plan,self.cells[m],trial,dict(commit='TOY',kind='ARTIFICIAL_ONLY'),'TOY')
            prices+=records
        records,models,pred=A.calibrate(prices,dict(commit='TOY',path='TOY'))
        self.assertEqual(pred,dict(X=dict(numerator=247,denominator=1),Y=dict(numerator=411,denominator=1)))
        self.assertEqual(models['X']['coefficients']['q'],dict(numerator=3,denominator=1))
        for mutate in (lambda p:p[0].update(m=4),lambda p:p[0].update(m=9),lambda p:p[0]['physical_context']['physical_candidate'].update(delta=.4),lambda p:p[0].update(axis='Y')):
            wrong=deepcopy(prices);mutate(wrong)
            with self.assertRaises((ValueError,KeyError)):A.calibrate(wrong,dict(commit='TOY',path='TOY'))


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CoarsePrices)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Previous.Metadata))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        import subprocess
        raise SystemExit(subprocess.call([sys.executable,'-B',str(ROOT/'tests/check_pf_m0_prices.py'),'--runtime']))
    raise SystemExit(1)
