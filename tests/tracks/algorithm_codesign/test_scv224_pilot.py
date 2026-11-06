"""Injected toy batches/prices only. Scientific services and RNG are trapped."""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from trottertracks.algorithm_codesign import a1_control_variates as cv
from trottertracks.algorithm_codesign import a1_n3_campaign as legacy
from trottertracks.algorithm_codesign import a1_n3_connection as n3
from trottertracks.algorithm_codesign import scv_scalar as scalar
from trottertracks.algorithm_codesign import scv224_pilot as a
from test_scv_scalar import metadata, batch, scope


class Collector:
    def __init__(self,out,control,mode=None):
        self.out=Path(out);self.control=control;self.mode=mode;self.calls=[];self.stops=[];self.finalized=0

    def collect(self,axis,stage,quotas):
        self.calls.append((axis,stage,quotas))
        if self.mode=='worker_failure':raise RuntimeError('invented worker failure')
        rows=tuple((h,batch(self.control,h,stage,n,axis,
            constant_feature=(self.mode=='rank'),independent_response=True)) for h,n in quotas)
        costs=tuple(cv.TrialCost(b.scope,r.stream_id,r.trial_id,stage,cv.FULL_SCOPE,
            (None,F(2),F(7),None,F(5)),F(0),F(0),1234) for _,b in rows for r in b.records)
        e=a.Envelope(a.CAMPAIGN_CANDIDATE,a.PURPOSE,self.control.metadata.identity,self.control.identity,
            axis,stage,rows,costs,sum(len(b.records) for _,b in rows))
        if self.mode=='campaign':e=replace(e,campaign='OLD752')
        if self.mode=='control':e=replace(e,control_identity='other')
        if self.mode=='metadata':e=replace(e,metadata_identity='other')
        if self.mode=='missing_price':e=replace(e,costs=e.costs[:-1])
        if self.mode=='unknown_price':e=replace(e,costs=(replace(costs[0],quantum=(None,2,None,None,5)),*costs[1:]))
        if self.mode=='duplicate_price':e=replace(e,costs=(*costs[:-1],costs[0]))
        if self.mode=='bad_stream':
            h,b=rows[0];b=replace(b,records=tuple(replace(r,stream_id='old-stream') for r in b.records))
            e=replace(e,rows=((h,b),*rows[1:]))
        if self.mode=='price_scope':e=replace(e,costs=(replace(costs[0],scope=replace(costs[0].scope,axis='Y')), *costs[1:]))
        return e

    def request_stop(self,reason):
        assert (self.out/'STOP_REQUESTED.json').exists()
        self.stops.append(reason)

    def finalize(self):
        assert self.stops and (self.out/'STOP_REQUESTED.json').exists()
        self.finalized+=1
        return dict(artificial=True,all_tails_included=True)


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.out=Path(self.tmp.name)
        md,*_=metadata();self.control=scalar.select_fragment_control(md)
        self.scopes={axis:scope(md,axis) for axis in ('X','Y')}
        legacy.save(self.out/'SOURCE.json',a.source_fields(a.CAMPAIGN_CANDIDATE,self.control))

    def driver(self,mode=None):
        c=Collector(self.out,self.control,mode)
        return a.Driver(self.out,a.CAMPAIGN_CANDIDATE,self.control,self.scopes,c),c

    def test_full_toy_224_paired_campaign_and_single_physical_charge(self):
        d,c=self.driver()
        with patch.object(n3,'load_certified_services',side_effect=AssertionError('science loader')), \
             patch.object(n3,'pcg64_streams',side_effect=AssertionError('scientific RNG')):
            r=d.run()
        self.assertEqual(r['status'],'COMPLETE',r)
        self.assertEqual([(x,y) for x,y,_ in c.calls],[('X','train'),('Y','train'),('X','prod'),('Y','prod')])
        costs=json.loads((self.out/'SCV_ACCOUNTING.json').read_text())
        self.assertEqual(costs['physical_shots_returned'],224);self.assertEqual(F(costs['physical_RZ']),224*7)
        self.assertEqual(F(costs['train_RZ']),64*7)
        self.assertEqual(F(costs['B1_OL_reference_evaluation_RZ']),160*7)
        self.assertEqual(F(costs['SCV_training_plus_evaluation_RZ']),224*7)
        self.assertEqual(costs['cache_hits_returned'],224)
        self.assertTrue(costs['cache_hits_are_charged']);self.assertEqual(c.finalized,1)
        self.assertTrue((self.out/'STOP.json').exists())
        for axis in ('X','Y'):
            self.assertEqual(r['axes'][axis]['precision'],'NOT_CERTIFIED')
            self.assertFalse(r['axes'][axis]['independent_method_campaigns'])

    def test_no_retry_repeated_run_or_driver(self):
        d,c=self.driver();d.run();before=len(c.calls)
        with self.assertRaisesRegex(ValueError,'single run'):d.run()
        with self.assertRaisesRegex(ValueError,'STOP exists'):self.driver()
        self.assertEqual(len(c.calls),before)

    def test_existing_stop_rejected_before_collector(self):
        legacy.save(self.out/'STOP.json',dict(old=True))
        with self.assertRaisesRegex(ValueError,'STOP exists'):self.driver()
        self.assertEqual(json.loads((self.out/'STOP.json').read_text()),dict(old=True))

    def test_existing_invocation_refuses_double_start(self):
        self.driver()
        with self.assertRaises(FileExistsError):self.driver()

    def test_wrong_source_and_axes_rejected_without_acquisition(self):
        saved=(self.out/'SOURCE.json').read_bytes()
        (self.out/'SOURCE.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'SOURCE identity'):self.driver()
        (self.out/'SOURCE.json').write_bytes(saved)
        c=Collector(self.out,self.control)
        with self.assertRaises(ValueError):a.Driver(self.out,a.CAMPAIGN_CANDIDATE,self.control,{'X':self.scopes['X']},c)
        self.assertFalse(c.calls)

    def test_source_changed_between_initialization_and_acquire_stops(self):
        d,c=self.driver();(self.out/'SOURCE.json').write_text('{}')
        r=d.run();self.assertEqual(r['status'],'STOPPED');self.assertFalse(c.calls)

    def test_wrong_provenance_stops_before_fit_preserves_received_bits(self):
        for mode in ('campaign','control','metadata','bad_stream','price_scope'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);legacy.save(out/'SOURCE.json',a.source_fields(a.CAMPAIGN_CANDIDATE,self.control))
                c=Collector(out,self.control,mode);d=a.Driver(out,a.CAMPAIGN_CANDIDATE,self.control,self.scopes,c)
                with patch.object(cv,'fit_and_freeze',side_effect=AssertionError('fit forbidden')):
                    r=d.run()
                self.assertEqual(r['status'],'STOPPED');self.assertEqual(len(c.calls),1)
                self.assertTrue((out/'RECEIVED_X_train.json').exists());self.assertEqual(c.finalized,1)

    def test_missing_or_duplicate_prices_unknown_and_no_new_measurement(self):
        for mode in ('missing_price','unknown_price','duplicate_price'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);legacy.save(out/'SOURCE.json',a.source_fields(a.CAMPAIGN_CANDIDATE,self.control))
                c=Collector(out,self.control,mode);d=a.Driver(out,a.CAMPAIGN_CANDIDATE,self.control,self.scopes,c)
                r=d.run();self.assertEqual(r['status'],'STOPPED');self.assertEqual(len(c.calls),1)
                self.assertTrue((out/'RECEIVED_X_train.json').exists())
                cost=json.loads((out/'SCV_ACCOUNTING.json').read_text())
                if mode!='duplicate_price':self.assertFalse(cost['cost_complete']);self.assertIsNone(cost['physical_RZ'])
                else:self.assertEqual(cost['status'],'UNKNOWN')

    def test_rank_failure_preserves_fit_then_stop_before_evaluation(self):
        d,c=self.driver('rank');r=d.run()
        self.assertEqual(r['status'],'STOPPED');self.assertEqual(len(c.calls),1)
        fit=json.loads((self.out/'X_FIT.json').read_text());self.assertEqual(fit['rank'],1)
        self.assertEqual(c.finalized,1)

    def test_worker_failure_stop_receipt_precedes_drain(self):
        d,c=self.driver('worker_failure');r=d.run()
        self.assertEqual(r['status'],'STOPPED');self.assertEqual(c.finalized,1)
        self.assertIn('worker failure',r['reason'])

    def test_finalizer_failure_persists_stop_and_received_data(self):
        d,c=self.driver();c.finalize=Mock(side_effect=RuntimeError('artificial drain failure'))
        r=d.run();self.assertEqual(r['status'],'STOPPED')
        self.assertIn('finalizer',r['reason']);self.assertTrue((self.out/'STOP.json').exists())
        self.assertTrue((self.out/'RECEIVED_Y_prod.json').exists())

    def test_source_mutation_during_last_acquisition_stops_after_bit_save(self):
        d,c=self.driver();original=c.collect
        def corrupt(axis,stage,quotas):
            e=original(axis,stage,quotas)
            if (axis,stage)==('Y','prod'):(self.out/'SOURCE.json').write_text('{}')
            return e
        c.collect=corrupt
        r=d.run();self.assertEqual(r['status'],'STOPPED')
        self.assertTrue((self.out/'RECEIVED_Y_prod.json').exists())
        self.assertNotIn('Y',r['axes'])

    def test_fixed_quotas_single_acquisition_claim(self):
        d,c=self.driver();d.acquire('X','train')
        with self.assertRaisesRegex(ValueError,'no retry'):d.acquire('X','train')
        with self.assertRaises(ValueError):d.acquire('X','cal')
        self.assertEqual(len(c.calls),1)

    def test_domains_distinct_tokens_12_and_no_generator_invocation(self):
        with patch.object(n3,'pcg64_streams',side_effect=AssertionError('no RNG')):
            plan=a.preparation_plan();ds=a.domains()
        self.assertEqual(len(ds),12);self.assertEqual(len({d.token for d in ds}),12)
        self.assertEqual(plan['child_streams'],24);self.assertFalse(plan['scientific_execution_enabled'])
        self.assertEqual(plan['total_words_cap'],224*65)
        self.assertEqual(sum(d.stage=='train' for d in ds),2)

    def test_cli_run_refused_before_any_scientific_function(self):
        root=next(p for p in Path(__file__).parents if (p/'PROJECT_MAP.md').exists())
        spec=importlib.util.spec_from_file_location('scv_info',root/'scripts/tracks/algorithm_codesign/prepare_scv224_pilot.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        with patch.object(n3,'pcg64_streams',side_effect=AssertionError('no RNG')), \
             patch.object(n3,'load_certified_services',side_effect=AssertionError('no loader')):
            with self.assertRaises(SystemExit) as e:mod.main(['--run'])
        self.assertEqual(e.exception.code,2)

    def test_legacy_collector_rejects_old_source_cache_or_stopped_job(self):
        raw=NS(out=self.out,sampler=self.control.metadata.sampler,trials=[],costs=[])
        stop=object();runtime=NS(out=self.out,stop=stop)
        with patch.object(legacy,'_STOP_EVENT',stop):
            a.LegacyCollector(raw,a.CAMPAIGN_CANDIDATE,self.control,runtime)
            (self.out/'cache').mkdir();(self.out/'cache'/'old').write_text('not reusable')
            with self.assertRaisesRegex(ValueError,'journal/cache'):a.LegacyCollector(raw,a.CAMPAIGN_CANDIDATE,self.control,runtime)
            (self.out/'cache'/'old').unlink();raw.trials.append('old')
            with self.assertRaisesRegex(ValueError,'empty acquisition'):a.LegacyCollector(raw,a.CAMPAIGN_CANDIDATE,self.control,runtime)

    def test_legacy_adapter_dispatches_only_train_h1_and_paired_evaluation(self):
        fake=Collector(self.out,self.control)
        raw=NS(out=self.out,sampler=self.control.metadata.sampler,trials=[],costs=[],classical={})
        def group(method,axis,stage,quotas):
            self.assertEqual(method,'A1' if stage=='train' else 'B1-OL')
            e=fake.collect(axis,stage,quotas)
            raw.costs.extend(e.costs)
            raw.trials.extend(dict(domain=n3.Domain(method,axis,stage,h),trial=r.trial_id,cache_hit=True)
                for h,b in e.rows for r in b.records)
            return e.rows,None
        raw.group=group;stop=object();runtime=NS(out=self.out,stop=stop)
        with patch.object(legacy,'_STOP_EVENT',stop):
            c=a.LegacyCollector(raw,a.CAMPAIGN_CANDIDATE,self.control,runtime)
            d=a.Driver(self.out,a.CAMPAIGN_CANDIDATE,self.control,self.scopes,c)
            self.assertEqual(len(d.acquire('X','train')[0][1].records),32)
            self.assertEqual(sum(len(b.records) for _,b in d.acquire('X','prod')),80)


if __name__=='__main__':unittest.main()
