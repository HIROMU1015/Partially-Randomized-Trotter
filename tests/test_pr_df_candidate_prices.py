"""Artificial design/model and already-completed-path resume; no compile."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_generic_df_prices',ROOT/'scripts/run_pr_df_candidate_prices.py')
E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E)


def config(Rs=None):
    Rs=Rs or [1]*13
    return dict(id='ARTIFICIAL_CONFIG',spec=dict(method='PR',split={},formula='2nd',finite_K=2,L_D=8),rounds=[
        dict(m=m,q_PF=2**m,R_outer=Rs[m],actual_delta_pf={'numerator':1,'denominator':8},physical_candidate={'id':'PHYSICAL_'+str(m)},structure=dict(signed_weights=[1.],allocation=[Rs[m]]),law_binding={'fixture':'LAW'})for m in range(13)])


class PriceTests(unittest.TestCase):
    def test_s2_actual_schedule_design_rank_and_changed_piece_needs(self):
        one=E.design(config());self.assertEqual(one['mode'],'FULL_AFFINE_Q_R1');self.assertEqual(one['calibration_rounds'],[0,6]);self.assertFalse(one['reasons'])
        rs=[max(1,2**m//2)for m in range(13)];general=E.design(config(rs));self.assertEqual(general['mode'],'FULL_AFFINE_QR');self.assertFalse(general['reasons']);self.assertEqual(general['holdout_round'],4)
        bad=E.design(config([2]*13));self.assertIn('RANK_DEFICIENT_HEAD_Q_AND_TAIL_QR_COEFFICIENTS',bad['reasons'])
        other=config();other['spec']['formula']='4th(m5_best)';self.assertTrue(E.design(other)['reasons'])
        other=config();other['rounds'][4]['actual_delta_pf']={'numerator':1,'denominator':16};self.assertIn('MULTIPLE_ACTUAL_STEPS_REQUIRE_SEPARATE_CALIBRATION_GROUPS',E.design(other)['reasons'])

    def test_axis_fit_freeze_and_exact_descriptive_holdout(self):
        cfg=config();cal=[dict(m=m,axis=a,q=2**m,R=1,mean_C=E.enc(11+5*2**m))for a in ('X','Y')for m in (0,6)]
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.df-price-test-')as d:
            out=Path(d);inputs=dict(configuration={'fixture':'CONFIG'},output_directory=E.rel(out),model_design=dict(mode='FULL_AFFINE_Q_R1',calibration_rounds=[0,6]))
            with patch.object(E,'read',return_value=cfg),patch.object(E,'means',return_value=cal):E.freeze(inputs,'ARTIFICIAL_RAW','ARTIFICIAL_SOURCE')
            model=json.loads((out/'FROZEN_MODELS.json').read_text());prices=json.loads((out/'AXIS_PRICES.json').read_text())
            self.assertEqual(model['models']['X']['alpha'],E.enc(11));self.assertEqual(model['models']['Y']['beta'],E.enc(5));self.assertEqual(len(prices['rows']),26)
            self.assertTrue(all(E.dec(r['C_ref'])>0 for r in prices['rows']))
            hold=[dict(m=3,axis=a,mean_C=E.enc(52))for a in ('X','Y')]
            with patch.object(E,'read',return_value=prices),patch.object(E,'means',return_value=hold):E.compare(inputs,'ARTIFICIAL_HOLD','ARTIFICIAL_MODEL','ARTIFICIAL_SOURCE')
            result=json.loads((out/'HOLDOUT_COMPARISON.json').read_text());self.assertEqual(result['rows'][0]['signed_error'],E.enc(-1));self.assertEqual(result['rows'][1]['signed_relative_error'],E.enc(Fraction(-1,52)))
            self.assertEqual(result['model_verdict'],'NOT_ASSESSED_NO_ADOPTED_TOLERANCE')

    def test_reuse_complete_receipt_and_prior_start_no_retry(self):
        fixture_spec=importlib.util.spec_from_file_location('_df_price_receipt_fixtures',ROOT/'tests/test_partial_df_price_reuse.py')
        fixtures=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixtures)
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.df-price-test-')as d:
            out=Path(d);base=E.rel(out);cell=fixtures.cell();inputs=dict(output_directory=base,cells=[cell])
            receipt=out/'calibration/acquisition/m0/p0/PATH_RECEIPT.json';receipt.parent.mkdir(parents=True)
            value=fixtures.fixture_path(cell);receipt.write_text(json.dumps(value));original=receipt.read_bytes()
            for a in value['record']['records']:
                (receipt.parent/({'X':'cosine','Y':'sine'}[a['axis']]+'.json')).write_text(json.dumps(dict(metrics=a['metrics'],fingerprint=a['fingerprint'],compiler='COMPILER')))
            with patch.object(E,'runtime',return_value=SimpleNamespace()),patch.object(E,'blob',return_value=b'ARTIFICIAL_SOURCE'):
                self.assertTrue(E.acquire(inputs,'INPUTS','calibration','SOURCE'))
            result=json.loads((out/'calibration/acquisition/RESULT.json').read_text());self.assertEqual(result['new_paths'],0);self.assertEqual(result['reused_axes'],2);self.assertEqual(receipt.read_bytes(),original)
            receipt.unlink();(out/'calibration/acquisition/STARTED_m0_p0.json').write_text('{}')
            with patch.object(E,'runtime',return_value=SimpleNamespace()),patch.object(E,'blob',return_value=b'ARTIFICIAL_SOURCE'):
                self.assertFalse(E.acquire(inputs,'INPUTS','calibration','SOURCE'))
            result=json.loads((out/'calibration/acquisition/RESULT.json').read_text());self.assertTrue(all(a['status']=='HELD_PRIOR_START_OR_FAILURE_NO_RETRY'for a in result['rows'][0]['axes'].values()));self.assertEqual(result['new_paths'],0)


if __name__=='__main__':unittest.main(verbosity=2)
