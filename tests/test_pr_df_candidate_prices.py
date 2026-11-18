"""Artificial design/model and already-completed-path resume; no compile."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import resource
import time
import traceback
from threading import Event
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

    def test_split_axis_jobs_keep_budget_stop_markers_and_one_event_backing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            cell=dict(m=3,request=dict(request_id='test',axes=['X','Y']))
            fresh={a:dict(status='MISSING') for a in ('X','Y')}
            jobs=E.pending_axis_jobs(cell,0,out,deepcopy(fresh),parallel=True)
            self.assertEqual([job[0]['request']['axes'] for job in jobs],[['X'],['Y']])
            self.assertEqual({job[1] for job in jobs},{0})
            self.assertEqual(cell['request']['axes'],['X','Y'])
            self.assertEqual(len({E.axis_stage_path(out/'STOP.json',selected,selected['request'])
                                  for selected,_ in jobs}),2)
            (out/'STARTED_m3_p0_axesX.json').touch()
            remaining=E.pending_axis_jobs(cell,0,out,deepcopy(fresh),parallel=True)
            self.assertEqual([job[0]['request']['axes'] for job in remaining],[['Y']])
            (out/'STARTED_m3_p0_axesXY.json').touch()
            self.assertFalse(E.pending_axis_jobs(cell,0,out,deepcopy(fresh),parallel=True))
            writer=E.shared_event_writer(E.write,lambda data:json.dumps(data,sort_keys=True))
            paths=[]
            for axis in ('X','Y'):
                p=out/('axes_'+axis)/'events_PRIVATE.json'
                p.parent.mkdir()
                writer(p,{'sealed_events':[1,2,3]})
                paths.append(p)
            self.assertTrue(paths[0].samefile(paths[1]))
            with self.assertRaisesRegex(ValueError,'event log differs'):
                writer(paths[1],{'sealed_events':[7]})

    def test_fixed_selected_acquirer_compiles_and_validates_both_individual_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'acquisition').mkdir()
            digest=lambda rows:json.dumps(rows,sort_keys=True)
            request=SimpleNamespace(rte_occurrence=SimpleNamespace(
                events=[SimpleNamespace(to_dict=lambda:{'kind':'ZZ'})]))
            ctx=SimpleNamespace(compile=lambda *args,**kwargs:None,prep=object(),
                h={'sample_requests':lambda *args:([request],[17],[])})
            science=SimpleNamespace(m=SimpleNamespace(digest=digest),write=E.write)
            wrapper=SimpleNamespace(num_qubits=9,num_clbits=1,count_ops=lambda:{'measure':1})
            axis_files={'X':'cosine','Y':'sine'}

            class Adapter:
                def __init__(self,*args,observer,**kwargs):
                    self.stage=observer

                def acquire(self,req,requests,*,cohort,index,directory):
                    axis,=req['axes']
                    directory.mkdir(parents=True)
                    event_rows=[[{'kind':'ZZ'}]]
                    science.write(directory/'events_PRIVATE.json',event_rows)
                    self.stage(axis+'_compile','STARTED')
                    ctx.compile(wrapper,'COMPILER')
                    E.write(directory/(axis_files[axis]+'.json'),dict(fingerprint=axis,compiler='COMPILER'))
                    return dict(seeds=[17],event_digest=digest(event_rows),new_compiles=1,
                                cache_hits=0,records=[dict(axis=axis,C=13,metrics=dict(rz_count=13),fingerprint=axis)])

            runtime=SimpleNamespace(science=science)
            helper=SimpleNamespace(ROOT=out,PUBLIC='',STATE=dict(runtimes={3:runtime},contexts={3:ctx},
                inputs=dict(cache_root=out/'cache')),time=time,json=json,resource=resource,traceback=traceback,
                FAILURE=Event(),P=SimpleNamespace(event_metrics=lambda events:events),write=E.write,
                D=SimpleNamespace(Adapter=Adapter,AXES=axis_files,p=SimpleNamespace(PROFILE={'compiler':'COMPILER'})))
            cell=dict(m=3,cohort='TEST',seeds=[[17]],request=dict(request_id='test',axes=['X','Y'],
                physical_context=dict(physical_candidate={})))
            jobs=E.pending_axis_jobs(cell,0,out,{a:dict(status='MISSING') for a in 'XY'},parallel=True)
            with patch.object(E,'runtime',return_value=helper):
                results=[E.selected_acquire_path(job) for job in jobs]
            self.assertEqual([r['status'] for r in results],['VALID','VALID'],results)
            self.assertEqual([r['compiles'][0]['axis'] for r in results],['X','Y'])
            self.assertEqual(results[0]['record']['event_digest'],results[1]['record']['event_digest'])
            self.assertFalse(helper.FAILURE.is_set())

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
