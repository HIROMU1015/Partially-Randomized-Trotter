"""Invented integer laws, deterministic word tapes and mock costs ONLY."""
from dataclasses import asdict, replace
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util
import json
import unittest
from unittest.mock import Mock, patch

from trottertracks.algorithm_codesign import a1_control_variates as cv
from trottertracks.algorithm_codesign import a1_n3_connection as a


def sampler():
    law = cv.OccurrenceLaw(cv.CategoricalLaw((0,2),(F(3,4),F(1,4)),'invented'),
        cv.CategoricalLaw(('a','b','c','zero'),(F(1,4),F(1,4),F(1,2),F(0)),'invented'))
    return a.ConditionalSampler(law,'invented','a','b')


def scopes():
    return {axis:cv.Scope('toy-target','invented','toy-full-wrapper','m0',axis,F(1)) for axis in ('X','Y')}


def tape(values):
    it = iter(values)
    return lambda: next(it)


def fixed_source(k=0):
    return lambda:k


def acquire(method,stage,cell,count=8, *, axis='X', scope=None, book=None):
    s = sampler(); domain = a.Domain(method,axis,stage,cell)
    scope = scope or scopes()[axis]
    book = book or a.Streams(64)
    book.bind(domain,fixed_source(),tape([0,a.M-1]*count))
    f = a.MeasurementFacade(lambda *args:F(1,2),scopes())
    return a.collect(f,s,book,scope,domain,count,lambda *args:NS(rz_count=7,cx_count=3),
        sampling_method='B1-OL' if method=='A1' and stage=='train' else None)


class N3Tests(unittest.TestCase):
    def test_direct_five_cells_match_exact_conditional_label_law(self):
        s = sampler()
        self.assertEqual(sum(p for _,p in s.cells('B1-OL')),1)
        for cell,p in s.cells('B1-OL'):
            for k in (0,1,a.M//2,a.M-2):
                path = s.draw('B1-OL',cell,tape([k]+[0]*16))
                self.assertEqual(s.cell_of('B1-OL',path),cell)
                self.assertNotIn('zero',path.labels)
        # Independent enumerated finite toy paths recover each cell's declared
        # mass. Conditional component choices use the same integer restrictions.
        sums = {k:F(0) for k,_ in s.cells('B1-OL')}
        for n,pn in zip(s.law.orders.labels,s.law.orders.masses):
            for labels in product(('a','b','c'),repeat=n+1):
                path = a.PathChoice(n,labels)
                mass = pn
                for label in labels:mass *= s.law.labels.mass(label)
                sums[s.cell_of('B1-OL',path)] += mass
        self.assertEqual(sums,dict(s.cells('B1-OL')))

    def test_actual_counts_no_float_renormalization_and_zero_support(self):
        s=sampler(); seen=[s._pick(s.law.labels,fixed_source(k)) for k in (0,a.M//4,a.M//2,a.M-1)]
        self.assertEqual(seen,['a','b','c','c'])
        with self.assertRaises(ValueError):s.draw('B1-OL','5',fixed_source())
        law=cv.OccurrenceLaw(cv.CategoricalLaw((0,2),(F(1,3),F(2,3)),'toy'),s.law.labels)
        with self.assertRaisesRegex(ValueError,'integer masses'):a.ConditionalSampler(law,'toy','a','b')

    def test_unbiased_integer_rejection_endpoint_and_word_cap(self):
        self.assertEqual(a.below(3,tape([a.M-1,4])),1)
        with self.assertRaises(ValueError):a.below(0,fixed_source())
        book=a.Streams(1);d=a.Domain('B0','X','cal','all');book.bind(d,fixed_source(a.M-1),fixed_source())
        word,_=book.claim(d,0)
        with self.assertRaisesRegex(ValueError,'budget exhausted'):a.below(3,word)
        for bad in (-1,a.M,True,.5):
            with self.assertRaises(ValueError):a.word53(fixed_source(bad))

    def test_facade_has_only_bit_metadata_no_probability_or_state_fields(self):
        probability=Mock(return_value=F(1,3));f=a.MeasurementFacade(probability,scopes());s=sampler()
        book=a.Streams(64);d=a.Domain('B0','Y','prod','all');book.bind(d,fixed_source(),fixed_source(a.M-1))
        record=f.acquire(s,book,d,0)
        self.assertEqual(record.bit,1);self.assertEqual(record.raw,-1);self.assertEqual(probability.call_count,1)
        self.assertEqual(set(asdict(record)),{'stream_id','trial_id','events','bit','weight','trajectory_id','wrapper_id'})
        self.assertFalse(hasattr(f,'probability'));self.assertFalse(hasattr(f,'state'))
        self.assertNotIn('probability',repr(f))

    def test_phase_is_metadata_not_double_applied_to_bit(self):
        path=a.PathChoice(2,('a','b','c'))
        self.assertEqual(path.event.phase,-1)
        m=cv.Measurement('s','t',(path.event,),0,F(1),'p','w');self.assertEqual(m.raw,1)

    def test_invalid_Born_value_stops_before_bit_draw(self):
        s=sampler();book=a.Streams(64);d=a.Domain('B0','X','prod','all');bit=Mock(return_value=0)
        book.bind(d,fixed_source(),bit);f=a.MeasurementFacade(lambda *args:F(2),scopes())
        with self.assertRaisesRegex(ValueError,'outside'):f.acquire(s,book,d,0)
        bit.assert_not_called()

    def test_stage_axis_method_stratum_and_trial_reuse_stops(self):
        book=a.Streams(64);d=a.Domain('B0','X','cal','all');w=fixed_source();b=fixed_source()
        book.bind(d,w,b);book.claim(d,0)
        with self.assertRaisesRegex(ValueError,'repeated trial'):book.claim(d,0)
        with self.assertRaisesRegex(ValueError,'reuse'):book.bind(a.Domain('B0','Y','prod','all'),w,fixed_source())
        with self.assertRaisesRegex(ValueError,'reuse'):book.bind(a.Domain('A1','X','prod','all'),b,b)

    def test_conditional_train_is_allowed_only_for_A1_training(self):
        s=sampler();f=a.MeasurementFacade(lambda *args:F(1,2),scopes());book=a.Streams(64)
        d=a.Domain('A1','X','prod','0');book.bind(d,fixed_source(),fixed_source())
        with self.assertRaisesRegex(ValueError,'only declared'):f.acquire(s,book,d,0,sampling_method='B1-OL')

    def test_conditional_training_full_rank_three_slopes_frozen(self):
        s=sampler();batches=[acquire('A1','train',cell)[0] for cell,_ in s.cells('B1-OL')]
        train=a.merge_training(batches);model,rank=a.freeze(s.bank,train,4)
        self.assertEqual(rank,4);self.assertEqual(len(model.coefficients),3)
        self.assertEqual(model.training,train)
        deficient=replace(train,records=train.records[:8])
        with self.assertRaisesRegex(ValueError,'rank deficient'):a.freeze(s.bank,deficient,4)
        with self.assertRaisesRegex(ValueError,'range guard'):a.freeze(s.bank,train,1)

    def test_variance_enclosure_is_positive_for_zero_observed_variance(self):
        c=a.calibrate([F(1)]*20,-1,1,F(1,100))
        self.assertEqual(c.sample_variance,0);self.assertGreater(c.variance_upper,0);self.assertLessEqual(c.variance_upper,1)
        with self.assertRaisesRegex(ValueError,'bounded'):a.calibrate([0,2],-1,1,F(1,100))

    def test_outward_shot_count_is_sufficient_for_both_selected_routes(self):
        import mpmath as mp
        c=a.calibrate([-1,1]*100,-1,1,F(1,100));n,r=a.sufficient_shots(c,F(1,4),F(1,1000),0,F(1,100))
        mp.mp.dps=100;e=mp.mpf(249)/1000;log=mp.log(200)
        self.assertGreaterEqual(r['Hoeffding'],2*log/e**2)
        self.assertGreaterEqual(r['Bernstein'],log*(2/e**2+4/(3*e)))
        self.assertEqual(n,min(r['Bernstein'],r['Hoeffding']))
        with self.assertRaisesRegex(ValueError,'headroom'):a.sufficient_shots(c,F(1,4),F(1,4),0,F(1,100))

    def test_confidence_union_all_methods_axes_and_positive_cells(self):
        alpha=F(1,20)
        self.assertEqual(sum(sum(a.confidence_cells(alpha,H))*H*2 for H in (1,2,5,1)),alpha)

    def test_full_mock_four_method_campaign_from_train_to_price_accounting(self):
        s=sampler();all_costs=[]
        for method in a.METHODS:
            train=[];model=None
            if method=='A1':
                for k,_ in s.cells('B1-OL'):
                    b,c=acquire(method,'train',k);train.append(b);all_costs.extend(c)
                model,_=a.freeze(s.bank,a.merge_training(train),4)
            elif method.startswith('B1'):
                for k,_ in s.cells(method):
                    b,c=acquire(method,'train',k);train.append(b);all_costs.extend(c)
            cal=[]
            for k,_ in s.cells(method):
                b,c=acquire(method,'cal',k);cal.append((k,b));all_costs.extend(c)
            plan=a.plan_production(s,method,cal,epsilon=F(1,2),engine_error=F(1,10000),alpha=F(1,10),cap=4096,
                pilot_prices=[F(7)]*len(cal),units=8,model=model,pilot_batches=() if model else tuple(train))
            prod=[]
            for k,p,n,e in plan.cells:
                b,c=acquire(method,'prod',k,n);prod.append((k,b));all_costs.extend(c)
            mean=a.production_mean(plan,s,prod);self.assertTrue(-2<=mean<=2)
            with self.assertRaises(ValueError):a.production_mean(plan,s,prod[:-1])
        categories={k:None for k in ('generation','fit','correction','compile','cache','engine_simulation','setup')}
        result=a.summarize_costs(all_costs,categories)
        self.assertEqual(result['quantum_total'][2],7*len(all_costs));self.assertIsNone(result['quantum_total'][0])
        self.assertIsNone(result['classical_online_seconds']);self.assertFalse(result['classical_cost_complete'])
        with self.assertRaisesRegex(ValueError,'unique'):a.summarize_costs(all_costs+all_costs[:1],categories)

    def test_allocation_retains_positive_strata_and_cap_is_fail_closed(self):
        c=a.calibrate([-1,1]*8,-1,1,F(1,100))
        _,es,ns=a.choose_allocation((F(1,4),F(3,4)),(c,c),(3,7),F(1,2),F(1,100),10000,units=8)
        self.assertTrue(min(ns)>0);self.assertLessEqual(sum(p*e for p,e in zip((F(1,4),F(3,4)),es)),F(1,2))
        with self.assertRaisesRegex(ValueError,'exceeds cap'):a.choose_allocation((F(1,4),F(3,4)),(c,c),(3,7),F(1,2),F(1,100),2,units=8)

    def test_publication_identity_mismatch_stops_before_real_decode(self):
        from trottertracks.algorithm_codesign import a1_engine_connection as old
        with patch.object(a,'inspect_certified_bundle',side_effect=ValueError('proof byte mismatch')),patch.object(old,'from_fixed_bundle',side_effect=AssertionError('no decode')):
            with self.assertRaisesRegex(ValueError,'byte mismatch'):a.load_certified_services('/invented/no/input')

    def test_production_cannot_reuse_calibration_or_change_method_domain(self):
        s=sampler();cal,cost=acquire('B0','cal','all')
        plan=a.plan_production(s,'B0',[('all',cal)],epsilon=1,engine_error=0,alpha=F(1,10),cap=1000,pilot_prices=[7],units=8)
        quota=plan.cells[0][2];prod,_=acquire('B0','prod','all',quota)
        a.production_mean(plan,s,[('all',prod)])
        wrong=replace(prod,records=tuple(replace(r,stream_id=cal.records[0].stream_id) for r in prod.records))
        with self.assertRaisesRegex(ValueError,'disjoint'):a.production_mean(plan,s,[('all',wrong)])
        other=replace(prod,records=tuple(replace(r,stream_id=a.Domain('A1','X','prod','all').token) for r in prod.records))
        with self.assertRaisesRegex(ValueError,'domain'):a.production_mean(plan,s,[('all',other)])

    def test_deferred_service_factory_uses_saved_certificate_without_requalification(self):
        from trottertracks.algorithm_codesign import a1_engine_connection as old
        from trottertracks.algorithm_codesign.a1_new_target_n1 import A,B
        binding=dict(N1_identities=dict(law_sha256='invented'),state_identity='toy-state',engine_certificate_sha256='toy-cert')
        law=dict(uniform_grid_size=a.M,application_label_order='products_then_rotation',orders=[0,2],order_masses=[dict(exact='3/4'),dict(exact='1/4')],
            label_masses=[dict(label=A,exact='1/4'),dict(label=B,exact='3/4')])
        proof=[dict(key=['sdg',[]],matrix_sha256='toy',matrix_complex_rationals=[[['1','0'],['0','0']],[['0','0'],['0','-1']]],error_upper='0')]
        certificate=dict(status='CONDITIONAL_UNIFORM_ENGINE_BOUND_CERTIFIED',state_identity='toy-state',max_gate_count=20)
        fake=NS(probability=Mock(return_value=F(1,2)))
        def read(p):return {'LAW.json':law,'ENGINE_ERROR_CERTIFIED.json':certificate,'GATE_CERTIFICATES.json':proof}[p.name]
        with patch.object(a,'inspect_certified_bundle',return_value=binding),patch.object(a,'verify_runtime'),patch.object(old,'from_fixed_bundle',return_value=(fake,binding)),patch.object(old,'read',side_effect=read),patch.object(old.SavedEngine,'qualify',side_effect=AssertionError('no qualification')):
            facade,prices,s,scope,cert=a.load_certified_services('/toy/path')
        self.assertEqual(cert,certificate);self.assertEqual(s.bank.law_id,'invented');fake.probability.assert_not_called()

    def test_changed_runtime_profile_stops_without_any_numerical_call(self):
        from trottertracks.algorithm_codesign import a1_engine_connection as old
        with (patch.object(old,'read',return_value=dict(source_sha256={},thread_environment={},python='old')),
              patch.object(old,'technical_profile',return_value=dict(source_sha256={},thread_environment={},python='changed'))):
            with self.assertRaisesRegex(ValueError,'runtime'):a.verify_runtime('/toy')

    def test_saved_proof_rehydrates_without_qualification_or_interval_calls(self):
        from trottertracks.algorithm_codesign import a1_engine_connection as old
        rows=[dict(key=['h',[]],matrix_sha256='invented',matrix_complex_rationals=[[['1','0'],['0','0']],[['0','0'],['1','0']]],error_upper='1/1024')]
        engine=NS()
        with patch.object(old,'certify_gate',side_effect=AssertionError('no qualification')),patch.object(old,'compose_error',side_effect=AssertionError('no error recalc')):
            a.attach_saved_proof(engine,rows,dict(status='CONDITIONAL_UNIFORM_ENGINE_BOUND_CERTIFIED',max_gate_count=4))
        self.assertEqual(engine.table['h',()]['matrix'][0][0],(F(1),F(0)))
        with self.assertRaisesRegex(ValueError,'duplicate'):a.attach_saved_proof(engine,rows*2,dict(status='x'))

    def test_mock_price_failure_retains_bit_in_sink_before_STOP(self):
        d=a.Domain('B0','X','cal','all');book=a.Streams(64);book.bind(d,fixed_source(),fixed_source());journal=[]
        f=a.MeasurementFacade(lambda *args:F(1,2),scopes())
        with self.assertRaisesRegex(ValueError,'missing price'):
            a.collect(f,sampler(),book,scopes()['X'],d,2,Mock(side_effect=ValueError('missing price')),sink=lambda *x:journal.append(x))
        self.assertEqual(len(journal),1);self.assertEqual(journal[0][0],'bit')

    def test_no_rng_real_input_compile_or_state_solve_called_by_mock_route(self):
        import numpy as np
        from trottertracks.algorithm_codesign import a1_engine_connection as old
        with patch.object(a,'load_certified_services',side_effect=AssertionError('real input')),patch.object(a,'pcg64_streams',side_effect=AssertionError('RNG')),patch.object(np.random,'PCG64',side_effect=AssertionError('RNG')),patch.object(old,'qualify_fixed_bundle',side_effect=AssertionError('qualify')),patch.object(np.linalg,'eigh',side_effect=AssertionError('eigh')):
            acquire('B0','cal','all')


class SavedKernelBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse existing 2-mode artificial fixture only; no fixed files loaded.
        path=Path(__file__).with_name('test_a1_engine_connection.py')
        spec=importlib.util.spec_from_file_location('invented_engine_fixture',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cls.fixture=module.EngineConnectionTests;cls.fixture.setUpClass()

    def test_existing_exact_engine_and_price_circuit_share_XY_and_cache_charges(self):
        from trottertracks.algorithm_codesign import a1_engine_connection as old
        from trottertracks.algorithm_codesign.shared_cpu import module
        engine=self.fixture().make_engine(); engine.qualify(0)
        label=next(iter(engine.index));path=a.PathChoice(0,(label,))
        compiler=NS(basis_gates=('rz','sx','x','cx'),backend_name=None,coupling_map=None,
            optimization_level=2,layout_method=None,routing_method=None,transpiler_seed=17,qiskit_version='1.3.0')
        provider=a.SavedPrices(engine);cache=NS(get_or_transpile=Mock(return_value=(NS(rz_count=11,cx_count=4),'key',True)))
        for axis in ('X','Y'):
            f=a.MeasurementFacade(engine.probability,scopes());s=sampler()
            # The facade path uses its invented law, so call private-kernel
            # bridge via a callback replacing only the invented label spelling.
            f=a.MeasurementFacade(lambda n,labels,b:engine.probability(0,(label,),b),scopes())
            book=a.Streams(64);d=a.Domain('B0',axis,'prod','all');book.bind(d,fixed_source(),fixed_source())
            record=f.acquire(s,book,d,0);self.assertIn(record.bit,(0,1))
            with patch.object(module('rte_compiled_cost'),'transpile_and_measure_cost',side_effect=AssertionError('no compile')):
                cost,hit=provider.price(path,axis,compiler,cache=cache)
            self.assertTrue(hit);self.assertEqual(cost.rz_count,11)
            circuit=cache.get_or_transpile.call_args.args[0]
            expected=engine.selected_circuit(0,(label,),axis)
            self.assertEqual([(i.operation.name,tuple(i.operation.params)) for i in circuit.data[:-1]],[(i.operation.name,tuple(i.operation.params)) for i in expected.data])
            self.assertEqual(circuit.data[-1].operation.name,'measure')


if __name__=='__main__':unittest.main()
