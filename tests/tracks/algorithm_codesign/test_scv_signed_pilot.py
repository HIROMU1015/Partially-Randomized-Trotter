"""Invented three-qubit metadata, deterministic toy bits, mock prices ONLY.

No H4 result is restored/evaluated; no science job/cron/real compile is started.
"""
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import nullcontext, ExitStack
from dataclasses import replace, FrozenInstanceError
from fractions import Fraction as F
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import json
import multiprocessing as mp
import os
import subprocess
import threading
import unittest
from unittest.mock import patch, Mock

from trottertracks.algorithm_codesign import scv_signed_restore as restore
from trottertracks.algorithm_codesign import scv_signed_pilot as p
from trottertracks.algorithm_codesign import scv_signed_runtime as r
from trottertracks.algorithm_codesign import scv_signed_activation as a
from test_scv_signed_support import fixture, scope, toy_batch
from test_a1_n3_campaign import fixture as worker_fixture, streams as tape


def saved_fixture(control):
    payload=dict(schema=restore.SCHEMA,feature_version=p.signed.FEATURE,metadata_identity=control.metadata.identity,
        control_identity=control.identity,metadata_structural_eligibility='ELIGIBLE',h1_nonconstant=True,
        feature_bounds_exact=['-1','1'],engine_loaded=False,state_loaded=False,old_measurement_data_used=False,
        next_stage_started=False,science_ready=False,snapshot=control.snapshot(),cells=[dict(cell=h,
            pattern=control.metadata.sampler.patterns[i],mass_exact=mass,mean_exact=control.means[i],
            second_moment_exact=control.second_moments[i],variance_exact=control.variances[i])
            for i,(h,mass) in enumerate(control.metadata.sampler.cells('B1-OL'))])
    data=p.c.dumps(payload).encode();md=control.metadata;base=md.base
    binding=restore.Binding('toy-commit','toy/METADATA_RESULT.json',p.scalar.digest(data),md.identity,control.identity,
        base.law_sha256,base.catalog_sha256,base.hamiltonian_hash,base.preparation_hash,md.sampler.a,md.sampler.b,md.system_qubits)
    return data,binding


class MockCollector:
    def __init__(self,out,control,gate):
        self.out,self.control,self.gate=Path(out),control,gate;self.calls=[];self.physical=0;self.raw=NS(classical={})
    def collect(self,axis,stage,quotas):
        self.calls.append((axis,stage,quotas));rows=[];costs=[]
        for h,n in quotas:
            batch=toy_batch(self.control,h,stage,n,axis=axis);rows.append((h,batch))
            for rec in batch.records:
                with self.gate.call('physical_trial'):self.physical+=1
                costs.append(p.cv.TrialCost(batch.scope,rec.stream_id,rec.trial_id,stage,p.cv.FULL_SCOPE,
                    (None,F(3),F(9),None,F(4)),F(0),F(0),123))
        return p.old.Envelope(self.gate.campaign,p.PURPOSE,self.control.metadata.identity,self.control.identity,
            axis,stage,tuple(rows),tuple(costs),len(costs))
    def request_stop(self,reason):self.gate.request_stop(reason)
    def finalize(self):
        assert (self.out/'STOP_REQUESTED.json').exists()
        return dict(mock=True,physical=self.physical)


def artificial_worker_init(out):
    out=Path(out);p.c._WORKER=(out,out/f'worker_{os.getpid()}.jsonl',None,None)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.out=Path(self.tmp.name)
        self.control,self.law,self.cat,self.kw=fixture();self.data,self.binding=saved_fixture(self.control)
        self.gate=r.CallGate(self.out,mp.get_context('fork').Event(),'invented-signed-campaign')
    def restore(self,data=None,binding=None,**overrides):
        return restore.restore_saved_control(self.data if data is None else data,self.law,self.cat,
            self.binding if binding is None else binding,commit='toy-commit',path='toy/METADATA_RESULT.json',**overrides)
    def driver(self,collector=None):
        p.c.save(self.out/'SOURCE.json',p.source_fields(self.gate.campaign,self.control))
        self.collector=collector or MockCollector(self.out,self.control,self.gate)
        return p.Driver(self.out,self.gate.campaign,self.control,{ax:scope(self.control,ax) for ax in ('X','Y')},self.collector,self.gate)
    def altered(self,fn):
        payload=json.loads(self.data);fn(payload);data=p.c.dumps(payload).encode()
        return data,replace(self.binding,result_sha256=p.scalar.digest(data))


class RestorationTests(Base):
    def test_restore_typed_without_constructor_or_signed_finite_sum(self):
        with patch.object(p.signed.SignedControl,'__post_init__',side_effect=AssertionError('no moment constructor')), \
             patch.object(p.signed.Metadata,'rotation',side_effect=AssertionError('no paths/moments')):
            control,masses=self.restore()
        self.assertIsInstance(control,p.signed.SignedControl);self.assertEqual(control.snapshot(),self.control.snapshot())
        self.assertEqual(masses,tuple(v for _,v in self.control.metadata.sampler.cells('B1-OL')))
        self.assertEqual(p.signed.ConditionalBank(control,'1').means,self.control.means[1:2])

    def test_restore_fit_paired_without_constructor(self):
        with patch.object(p.signed.SignedControl,'__post_init__',side_effect=AssertionError('no recalc')):
            control,_=self.restore();model=p.signed.fit_h1(control,toy_batch(control,'1','train',64))
            rows=tuple((h,toy_batch(control,h,'prod',n)) for h,n in zip(p.scalar.CELLS,p.COUNTS))
            p.signed.paired_evaluate(model,rows,counts=p.COUNTS)

    def test_hash_commit_path_reject(self):
        with self.assertRaises(ValueError):self.restore(self.data+b' ')
        for k in ('result_commit','result_path'):
            with self.assertRaises(ValueError):self.restore(binding=replace(self.binding,**{k:'other'}))

    def test_bad_fractions_schema_feature_fields_identity_reject(self):
        mutations=[lambda x:x.update(schema='wrong'),lambda x:x.update(feature_version='indicator'),
            lambda x:x.update(metadata_identity='bad'),lambda x:x.update(control_identity='bad'),
            lambda x:x['cells'].pop(),lambda x:x['cells'][1].update(mean_exact='nan'),
            lambda x:x['cells'][1].update(mean_exact='1/0'),lambda x:x['cells'][1].update(mean_exact=1),
            lambda x:x['cells'][1].update(mean_exact='2/6'),lambda x:x['cells'][1].update(variance_exact='0'),
            lambda x:x['snapshot'].update(conditional_means=['0']*5),lambda x:x.update(h1_nonconstant=False),
            lambda x:x['cells'][1].update(pattern=[0,0,0,0])]
        for fn in mutations:
            data,b=self.altered(fn)
            with self.subTest(data=data[:20]),self.assertRaises((ValueError,TypeError)):self.restore(data,b)

    def test_wrong_law_catalog_sign_support_scope_reject(self):
        for k in ('law_sha256','catalog_sha256','hamiltonian_hash','preparation_hash','a','b','system_qubits'):
            value=2 if k=='system_qubits' else 'other'
            with self.subTest(field=k),self.assertRaises(ValueError):self.restore(binding=replace(self.binding,**{k:value}))

    def test_control_immutable_and_sampler_tamper_detected(self):
        control,_=self.restore()
        with self.assertRaises(FrozenInstanceError):control.means=(0,)*5
        control.metadata.sampler.a='b'
        with self.assertRaisesRegex(ValueError,'identity changed'):control.value(p.n3.PathChoice(0,('a',)).event)

    def test_duplicate_json_field_rejected(self):
        data=self.data[:-1]+b',"schema":"bad"}'
        with self.assertRaisesRegex(ValueError,'duplicate'):self.restore(data,replace(self.binding,result_sha256=p.scalar.digest(data)))


class PilotTests(Base):
    def test_448_stage_order_independent_fit_full_half_and_paid_training(self):
        driver=self.driver();real=p.signed.fit_h1;order=[]
        def fit(control,batch,**kw):
            self.assertEqual([(a,b) for a,b,_ in self.collector.calls],[('X','train'),('Y','train')])
            order.append(batch.scope.axis);return real(control,batch,**kw)
        with patch.object(p.signed,'fit_h1',side_effect=fit) as fits,patch.object(p.signed,'paired_evaluate',wraps=p.signed.paired_evaluate) as paired:
            result=driver.run()
        self.assertEqual(result['status'],'COMPLETE',result['reason']);self.assertEqual(order,['X','Y'])
        self.assertEqual(fits.call_count,2);self.assertEqual(paired.call_count,6)
        self.assertEqual(self.collector.physical,448);self.assertEqual(len(driver.seen),448)
        ledger=json.loads((self.out/'SIGNED_ACCOUNTING.json').read_text())
        self.assertEqual(ledger['physical_RZ'],'4032');self.assertEqual(ledger['train_RZ'],'1152')
        self.assertEqual(ledger['B1_OL_reference_evaluation_RZ'],'2880');self.assertEqual(ledger['cache_hits_returned'],448)
        self.assertTrue((self.out/'STOP.json').exists())
        with self.assertRaisesRegex(ValueError,'single invocation'):driver.run()

    def test_no_evaluation_before_both_fit(self):
        d=self.driver();d.next_acquire=2
        with self.assertRaisesRegex(ValueError,'both fits'):d.acquire('X','prod')
        self.assertEqual(self.collector.calls,[])

    def test_failed_X_guard_keeps_both_train_costs_no_Y_fit_eval(self):
        d=self.driver()
        with patch.object(p,'signed_guard',side_effect=ValueError('guard')) as guard:
            result=d.run()
        self.assertEqual(result['status'],'STOPPED');self.assertEqual(guard.call_count,1)
        self.assertEqual(self.collector.physical,128);self.assertEqual(set(d.models),set())
        ledger=json.loads((self.out/'SIGNED_ACCOUNTING.json').read_text())
        self.assertEqual(ledger['train_RZ'],'1152');self.assertFalse(ledger['complete_campaign'])
        self.assertIn('UNKNOWN',ledger['physical_RZ'])

    def test_rank_deficient_global_stop_no_refit(self):
        d=self.driver();real=self.collector.collect
        def collect(axis,stage,q):
            e=real(axis,stage,q)
            batch=toy_batch(self.control,'1',stage,64,axis=axis,constant_feature=True)
            return replace(e,rows=(('1',batch),))
        self.collector.collect=collect
        with patch.object(p.cv,'fit_and_freeze',wraps=p.cv.fit_and_freeze) as fit:
            result=d.run()
        self.assertEqual(result['status'],'STOPPED');self.assertEqual(fit.call_count,1)
        self.assertEqual(self.collector.physical,128)

    def test_nonfinite_OLS_no_rescue(self):
        import numpy as np
        d=self.driver()
        with patch.object(np.linalg,'lstsq',return_value=(np.array([0.,float('nan')]),None,None,None)) as solve:
            result=d.run()
        self.assertEqual(solve.call_count,1);self.assertIn('nonfinite',result['reason']);self.assertEqual(self.collector.physical,128)

    def test_signed_guard_exact_boundary_6_and_above(self):
        model=p.signed.fit_h1(self.control,toy_batch(self.control,'1','train',64));active=model.models[1]
        for slope,passed in ((F(2),True),(F(-2),True),(F(2)+F(1,2**53),False)):
            frozen=replace(active.frozen,coefficients=(slope,));other=replace(model,models=(model.models[0],replace(active,frozen=frozen),*model.models[2:]))
            if passed:self.assertEqual(p.signed_guard(other)['actual_width'],6)
            else:
                with self.assertRaises(ValueError):p.signed_guard(other)

    def test_half_partition_trial_ids_not_completion_order(self):
        rows=tuple((h,toy_batch(self.control,h,'prod',n)) for h,n in zip(p.scalar.CELLS,p.COUNTS))
        shuffled=tuple((h,replace(b,records=b.records[::-1])) for h,b in rows)
        ha,hb=p.split_rows(shuffled)
        for (_,x),(_,y),n in zip(ha,hb,p.COUNTS):
            self.assertEqual([int(v.trial_id) for v in x.records],list(range(n//2)))
            self.assertEqual([int(v.trial_id) for v in y.records],list(range(n//2,n)))
            self.assertFalse(set(v.key for v in x.records)&set(v.key for v in y.records))

    def test_full_half_missing_duplicate_cell_or_trial_reject(self):
        rows=tuple((h,toy_batch(self.control,h,'prod',n)) for h,n in zip(p.scalar.CELLS,p.COUNTS))
        for bad in (rows[:-1],(rows[0],rows[0],*rows[2:])):
            with self.assertRaises(ValueError):p.split_rows(bad)
        h,b=rows[0];bad=replace(b,records=(replace(b.records[0],trial_id='999'),*b.records[1:]))
        with self.assertRaises(ValueError):p.split_rows(((h,bad),*rows[1:]))

    def test_foreign_campaign_control_or_missing_price_stops(self):
        d=self.driver();real=self.collector.collect
        self.collector.collect=lambda *args:replace(real(*args),campaign='old-campaign')
        result=d.run();self.assertEqual(result['status'],'STOPPED');self.assertIn('foreign campaign',result['reason'])
        self.assertTrue((self.out/'RECEIVED_X_train.json').exists())

    def test_existing_stop_source_mismatch_no_acquisition(self):
        d=self.driver();(self.out/'STOP_REQUESTED.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'STOP'):d.acquire('X','train')
        self.assertEqual(self.collector.physical,0)

    def test_diagnostic_flags_not_CI(self):
        full=dict(B1_OL={'estimated_variance_of_mean':F(2)},SCV={'estimated_variance_of_mean':F(1)},
            cells=[{},dict(raw_all_equal=False,residual_all_equal=False,feature_counts={'-1':5,'1':5})])
        a0=full|dict(SCV={'estimated_variance_of_mean':F(3)})
        result=p.diagnostic(full,a0,full)
        self.assertIn('INCONCLUSIVE_SPLIT_DIAGNOSTIC',result['flags']);self.assertFalse(result['half_is_CI'])
        self.assertEqual(result['statistical_significance'],'NOT_ASSESSED')


class RuntimeTests(Base):
    def test_all_declared_caps_reject_next_before_body(self):
        self.assertEqual(p.CAPS['word'],29120);self.assertEqual(p.CAPS['apply_exact'],427840)
        for name in p.CAPS:
            out=self.out/name;out.mkdir();g=r.CallGate(out,mp.get_context('fork').Event(),'toy',caps={name:2})
            for _ in range(2):
                with g.call(name):pass
            with self.assertRaisesRegex(RuntimeError,'cap'):
                with g.call(name):self.fail('body beyond cap')
            self.assertEqual(g.snapshot()['observed'][name]['started'],2)

    def test_stop_start_race_inflight_return_and_short_lock(self):
        g=self.gate;entered=threading.Barrier(3);release=threading.Event()
        def task():
            with g.call('probability'):
                entered.wait(timeout=10);release.wait(timeout=10)
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(task) for _ in range(2)];entered.wait(timeout=10)
            g.request_stop('race');self.assertEqual(g.snapshot()['observed']['probability']['in_flight'],2)
            with self.assertRaises(RuntimeError):
                with g.call('price'):self.fail('new price after stop')
            release.set()
            for f in futures:f.result(timeout=10)
        events=[json.loads(x) for x in g.log.read_text().splitlines()];index=next(i for i,x in enumerate(events) if x['event']=='STOP_REQUESTED')
        self.assertFalse(any(x['event']=='CALL_START' for x in events[index+1:]))
        self.assertEqual(g.snapshot()['observed']['probability']['returned'],2)

    def test_truncated_return_is_unknown_not_shared_counter_success(self):
        with self.gate.call('price'):pass
        lines=self.gate.log.read_bytes().splitlines(keepends=True)
        self.gate.log.write_bytes(b''.join(lines[:-1])+lines[-1][:20])
        result=self.gate.snapshot()
        self.assertEqual(result['shared_counters']['price']['returned'],1)
        self.assertEqual(result['observed']['price']['returned'],0)
        self.assertEqual(result['observed']['price']['in_flight'],1)
        self.assertIsNotNone(result['incomplete_adapter_tail'])

    def test_killed_lock_owner_does_not_block_durable_stop(self):
        import signal
        ctx=mp.get_context('fork');ready=ctx.Event()
        def child():
            with self.gate.critical():
                ready.set();signal.pause()
        child_process=ctx.Process(target=child);child_process.start()
        try:
            self.assertTrue(ready.wait(10));os.kill(child_process.pid,signal.SIGKILL);child_process.join(10)
            self.assertEqual(child_process.exitcode,-signal.SIGKILL)
            self.gate.request_stop('artificial lock owner killed')
            self.assertTrue((self.out/'STOP_REQUESTED.json').exists())
        finally:
            if child_process.is_alive():os.kill(child_process.pid,signal.SIGKILL);child_process.join()

    def test_cross_process_stop_linearizes_with_parent_start(self):
        ctx=mp.get_context('fork');gate=self.gate;ready=ctx.Event();done=ctx.Event()
        def child():
            with gate.call('probability'):ready.set();done.wait(10)
        child_process=ctx.Process(target=child);child_process.start()
        try:
            self.assertTrue(ready.wait(10));gate.request_stop('parent');done.set();child_process.join(10)
            self.assertEqual(child_process.exitcode,0)
            with self.assertRaises(RuntimeError):
                with gate.call('compile'):pass
        finally:
            if child_process.is_alive():child_process.terminate();child_process.join()

    def test_prepare_replay_and_stop_before_new_path_claim(self):
        domain=p.n3.Domain('A1','X','train','1');book=p.n3.Streams(64);book.bind(domain,lambda:0,lambda:0)
        prepared=r.prepare_adapter(p.c.prepare_trial,self.gate)
        job=prepared(self.control.metadata.sampler,r.CountedStreams(book,self.gate),domain,0,self.out/'JOURNAL.jsonl',self.out/'words')
        self.assertEqual(job['domain'],domain)
        with self.assertRaisesRegex(ValueError,'replay'):prepared(self.control.metadata.sampler,book,domain,0,self.out/'JOURNAL.jsonl',self.out/'words')
        self.gate.request_stop('halt')
        with self.assertRaises(RuntimeError):prepared(self.control.metadata.sampler,book,domain,1,self.out/'JOURNAL.jsonl',self.out/'words')
        self.assertEqual(len((self.out/'GLOBAL_TRIALS.jsonl').read_text().splitlines()),1)

    def test_real_frozen_worker_with_toy_probability_keeps_bit_first_and_tails(self):
        services=worker_fixture();domain=p.n3.Domain('A1','X','train','1')
        job=r.prepare_adapter(p.c.prepare_trial,self.gate)(services[2],tape(domain),domain,0,self.out/'JOURNAL.jsonl',self.out/'engine_internal_words.jsonl')
        services[1].price.side_effect=ValueError('toy price failure')
        with patch.object(p.c,'_SERVICES',services),patch.object(p.c,'_WORKER',(self.out,self.out/'worker_toy.jsonl',None,None)),\
             patch.object(p.c,'_STOP_EVENT',self.gate.stop),r.connection(self.gate):
            result=r.worker_trial(job)
        self.assertTrue(result['bit_saved']);self.assertTrue(self.gate.stop.is_set())
        states=r.Runtime(self.out,self.gate).classify();self.assertEqual(states[job['ticket']],'BIT_DURABLE_PRICE_MISSING')
        self.assertEqual(p.c.reconcile(self.out)['A1']['unpriced_bits'],1)
        events=[json.loads(x)['event'] for x in (self.out/'worker_toy.jsonl').read_text().splitlines()]
        self.assertLess(events.index('BIT_SAVED'),events.index('WORKER_STOP'))

    def test_partial_collector_retains_unknown_unstarted_and_unpriced_tails(self):
        services=worker_fixture();domain=p.n3.Domain('A1','X','train','1')
        job=r.prepare_adapter(p.c.prepare_trial,self.gate)(services[2],tape(domain),domain,0,self.out/'JOURNAL.jsonl',self.out/'engine_internal_words.jsonl')
        services[1].price.side_effect=ValueError('toy price failure')
        with patch.object(p.c,'_SERVICES',services),patch.object(p.c,'_WORKER',(self.out,self.out/'worker_toy.jsonl',None,None)),\
             patch.object(p.c,'_STOP_EVENT',self.gate.stop),r.connection(self.gate):r.worker_trial(job)
        # Another accepted path never reached a worker. Do not omit its UNKNOWN cost.
        p.c.append(self.out/'JOURNAL.jsonl',dict(event='TRIAL_PLANNED',ticket='unstarted',domain=domain,trial=1))
        collector=object.__new__(r.Collector);collector.raw=NS(out=self.out,pool=Mock(),classical={})
        collector.campaign=self.gate.campaign;collector.gate=self.gate;collector.runtime=r.Runtime(self.out,self.gate)
        collector.finalize()
        record=json.loads((self.out/'SIGNED_TAIL_ACCOUNTING.json').read_text())
        self.assertEqual(record['total_RZ'],'UNKNOWN');self.assertEqual(record['known_RZ'],0)
        self.assertEqual(record['trials'][1]['classification'],'PLANNED_NO_DURABLE_WORKER_START')

    def test_each_numeric_provider_guard_calls_same_body_once(self):
        owner=NS(method=Mock(return_value='opaque result'))
        original=owner.method
        with ExitStack() as stack:
            r._observe(stack,self.gate,owner,'method','price',local={'ticket':'toy'})
            self.assertEqual(owner.method(),'opaque result')
            self.gate.request_stop('toy stop')
            with self.assertRaises(RuntimeError):owner.method()
        self.assertEqual(original.call_count,1)

    def test_worker_gate_limit_955_before_the_next_body(self):
        local={'gates':955,'ticket':'toy'};owner=NS(method=Mock())
        with ExitStack() as stack:
            r._observe(stack,self.gate,owner,'method','apply_exact',local=local)
            with self.assertRaisesRegex(RuntimeError,'955'):owner.method()
        self.assertEqual(local['gates'],956)
        self.assertEqual(self.gate.snapshot()['observed']['apply_exact']['started'],0)

    def test_fixed_53bit_stream_mapping_and_separation_on_artificial_fixture(self):
        ds=p.domains();self.assertEqual(len(ds),12);self.assertEqual(p.SEED,20261110)
        book=p.n3.pcg64_streams(p.SEED,ds,path_word_cap=64)  # invented campaign only
        seen=[]
        for d in ds:
            path,bit=book.claim(d,0);words=(path(),bit());seen.extend(words)
            with self.assertRaises(ValueError):book.claim(d,0)
        self.assertEqual(len(set(seen)),24)
        self.assertEqual([x['path_spawn_key'] for x in p.settings()['domain_mapping']],[[2*i] for i in range(12)])
        self.assertEqual([x['bit_spawn_key'] for x in p.settings()['domain_mapping']],[[2*i+1] for i in range(12)])

    def test_word_limit_64plus1_and_invalid_word_fail_closed(self):
        d=p.domains()[0];book=p.n3.Streams(64);book.bind(d,lambda:0,lambda:0)
        path,bit=book.claim(d,0)
        for _ in range(64):path()
        bit()
        with self.assertRaises(ValueError):path()
        with self.assertRaises(ValueError):bit()

    def test_worker_attempt_duplicate_rejected_before_second_measurement(self):
        domain=p.domains()[0];slot=self.gate.trial_slot(domain,0);self.gate.planned[slot]=1
        ticket=__import__('hashlib').sha256((domain.token+':0').encode()).hexdigest()
        job=dict(domain=domain,trial=0,ticket=ticket)
        self.gate.accept_attempt(job)
        with self.assertRaisesRegex(ValueError,'replay'):self.gate.accept_attempt(job)
        self.assertTrue(self.gate.stop.is_set())

    def test_existing_group_managed_fork_workers_share_gate_and_keep_bit_price(self):
        services=worker_fixture();domain=p.n3.Domain('A1','X','train','1');book=p.n3.Streams(64)
        book.bind(domain,lambda:0,lambda:0);life=r.Runtime(self.out,self.gate)
        with patch.object(p.c,'_SERVICES',services),patch.object(p.c,'_STOP_EVENT',self.gate.stop),r.connection(self.gate):
            pool=r.managed.ManagedPool(life,max_workers=2,mp_context=mp.get_context('fork'),
                initializer=artificial_worker_init,initargs=(self.out,))
            try:
                raw=p.c.Campaign(self.out,services,r.CountedStreams(book,self.gate),pool,2)
                rows,prices=raw.group('A1','X','train',[('1',2)])
                self.assertEqual(len(rows[0][1].records),2);self.assertEqual(prices,(F(11),))
            finally:
                life.request_stop('toy end');pool.shutdown()
        methods=p.c.reconcile(self.out)
        self.assertEqual(methods['A1']['bits_saved'],2);self.assertEqual(methods['A1']['realized_RZ_priced'],22)
        self.assertEqual(self.gate.snapshot()['observed']['physical_trial']['returned'],2)
        self.assertTrue(all(v['exit_code']==0 for v in json.loads((self.out/'WORKER_EXIT_CODES.json').read_text()).values()))

    def test_managed_pool_cancel_stop_receipt_before_drain(self):
        life=r.Runtime(self.out,self.gate);fake=Mock();future=Future();fake.submit.return_value=future;fake._processes={}
        with patch.object(r.managed,'ProcessPoolExecutor',return_value=fake):
            pool=r.managed.ManagedPool(life,max_workers=2);pool.submit(lambda:0)
            life.request_stop('artificial stop');pool.shutdown()
            with self.assertRaises(RuntimeError):pool.submit(lambda:0)
        fake.shutdown.assert_called_once_with(wait=True,cancel_futures=True)
        self.assertTrue((self.out/'STOP_REQUESTED.json').exists())


class ActivationTests(Base):
    def test_manager_routing_is_existing_and_globals_restored(self):
        original=a.old.OUTPUT
        with patch.object(a.old,'manager',side_effect=lambda path:(a.old.OUTPUT,a.old.ENTRY,path)):
            self.assertEqual(a.manager('toy'),(a.OUTPUT,a.ENTRY,'toy'))
        self.assertEqual(a.old.OUTPUT,original)

    def test_consumed_dispatch_refuses_second_launch(self):
        control=self.out/a.CONTROL;control.mkdir(parents=True);p.c.save(control/'CONSUMED.json',{})
        with patch.object(a,'read_spec',return_value=(self.out,{})),patch.object(a.old.cron,'record_manager') as launch:
            self.assertEqual(a.dispatch('toy'),72)
        launch.assert_not_called()

    def test_foreign_manager_and_restart_rejected(self):
        path=self.out/a.CONTROL/'SPEC.json';path.parent.mkdir(parents=True)
        for kind,restart in (('old',False),(a.KIND,True)):
            p.c.save(path,dict(root=str(self.out),kind=kind,restart=restart))
            with self.assertRaisesRegex(ValueError,'manager'):a.read_spec(path)

    def test_existing_output_stop_forbids_parent_before_input(self):
        out=self.out/a.OUTPUT;out.mkdir(parents=True);(out/'STOP.json').write_text('{}')
        with patch.object(a,'read_spec',return_value=(self.out,{})),patch.object(a,'verify') as verify:
            with self.assertRaises(FileExistsError):a.parent('toy')
        verify.assert_not_called()

    def test_observed_memory_affinity_task_limit_not_arbitrary_low_cap(self):
        snap={'meminfo':'MemAvailable: 100000000 kB\n','pressure/memory':'some avg10=0.00\nfull avg10=0.00\n'}
        self.assertEqual(a.memory_admission(200,snap,per_worker_bytes=1000),160)
        snap['ancestor_cgroups']={'own':{'memory.max':'8000','memory.current':'2000'}}
        self.assertEqual(a.memory_admission(200,snap,per_worker_bytes=1000),6)
        snap['pressure/memory']='full avg10=0.01\n'
        with self.assertRaisesRegex(RuntimeError,'pressure'):a.memory_admission(4,snap,per_worker_bytes=1)

    def test_minimal_cron_environment_cli_help_passive(self):
        root=next(x for x in Path(__file__).parents if (x/'PROJECT_MAP.md').exists())
        env={k:os.environ[k] for k in ('HOME','USER','LOGNAME','SHELL','PATH','LANG') if k in os.environ}
        res=subprocess.run([a.old.PYTHON,'-B',str(root/a.ENTRY),'--help'],cwd='/tmp',env=env,capture_output=True,text=True)
        self.assertEqual(res.returncode,0,res.stderr);self.assertIn('--submit',res.stdout)

    def test_mock_parent_complete_448_single_restore_decode_stream_book(self):
        out=self.out/a.OUTPUT;out.mkdir(parents=True)
        spec=dict(source_commit='toy')
        scopes={ax:scope(self.control,ax) for ax in ('X','Y')}
        services=(None,None,self.control.metadata.sampler,scopes,{})
        def collector(raw,campaign,control,life,gate):
            return MockCollector(out,control,gate)
        with patch.object(a,'read_spec',return_value=(self.out,spec)),patch.object(a,'verify',return_value={}),\
             patch.object(p.c,'forbidden_science',return_value=nullcontext()),patch.object(a.admission,'wait_for_resources',return_value=3),\
             patch.object(a,'memory_admission',side_effect=lambda n,*args,**kw:n),\
             patch.object(a,'restore_once',return_value=(self.control,(F(1,5),)*5)) as restored,\
             patch.object(a.n3,'load_certified_services',return_value=services) as decoded,\
             patch.object(a.n3,'pcg64_streams') as rng,patch.object(a.managed,'ManagedPool'),\
             patch.object(a.runtime,'Collector',side_effect=collector):
            self.assertEqual(a.parent('toy'),0)
        self.assertEqual(restored.call_count,1);self.assertEqual(decoded.call_count,1)
        rng.assert_called_once_with(20261110,p.domains(),path_word_cap=64)
        self.assertEqual(json.loads((out/'SIGNED_ACCOUNTING.json').read_text())['physical_shots_returned'],448)
        self.assertEqual(json.loads((out/'BOUNDED_CALL_COUNTS.json').read_text())['observed']['OLS']['started'],2)

    def test_spec_source_profile_settings_mismatch_before_input(self):
        root=self.out;doc=root/a.DOC;doc.mkdir(parents=True)
        p.c.save(doc/'PROFILE_LOCK.json',{'toy':'profile'})
        p.c.save(doc/'INPUT_BINDING.json',{'toy':'binding'})
        p.c.save(doc/'SETTINGS.json',p.settings())
        git=lambda root,*args:'source' if args[0]=='rev-parse' else ''
        spec=dict(kind=a.KIND,campaign=a.CAMPAIGN,seed=a.SEED,output=str(root/a.OUTPUT),restart=False,
            source_commit='source',manager_SOURCE='5cb542ece4d208878a1dfc7011007b99297a5a55',python=a.old.PYTHON,source_remote_verified='source',binding={'toy':'binding'},profile_sha256=p.scalar.digest((doc/'PROFILE_LOCK.json').read_bytes()))
        with patch.object(p.c.kernel.n2,'git',side_effect=git),patch.object(a,'profile',return_value={'toy':'profile'}),\
             patch.object(a.n3,'verify_runtime'),patch.object(a,'input_binding',return_value={'toy':'binding'}):
            a.verify(root,spec)
            for change in (dict(source_commit='old'),dict(seed=1),dict(binding={}),dict(profile_sha256='bad')):
                with self.assertRaises(ValueError):a.verify(root,spec|change)
            with patch.object(a,'profile',return_value={'toy':'compiler changed'}):
                with self.assertRaisesRegex(ValueError,'compiler'):a.verify(root,spec)

    def test_mock_parent_resource_failure_is_zero_restore_decode_stream(self):
        out=self.out/a.OUTPUT;out.mkdir(parents=True)
        with patch.object(a,'read_spec',return_value=(self.out,dict(source_commit='toy'))),patch.object(a,'verify',return_value={}),\
             patch.object(p.c,'forbidden_science',return_value=nullcontext()),patch.object(a.admission,'wait_for_resources',side_effect=RuntimeError('toy resources')),\
             patch.object(a,'restore_once') as rest,patch.object(a.n3,'load_certified_services') as decode,patch.object(a.n3,'pcg64_streams') as rng:
            self.assertEqual(a.parent('toy'),1)
        rest.assert_not_called();decode.assert_not_called();rng.assert_not_called()
        self.assertTrue((out/'STOP.json').exists())


if __name__=='__main__':unittest.main()
