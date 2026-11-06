"""Invented LAW/catalog, deterministic metadata and bit tables ONLY."""
from collections import Counter
from dataclasses import replace, FrozenInstanceError
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from statistics import variance
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import copy
import json
import unittest
from unittest.mock import patch

from trottertracks.algorithm_codesign import scv_signed_support as a
from trottertracks.algorithm_codesign import a1_control_variates as cv
from trottertracks.algorithm_codesign import a1_n3_connection as n3
from trottertracks.algorithm_codesign import scv_scalar as old
from trottertracks.algorithm_codesign import scv224_pilot as accounting
from trottertracks.algorithm_codesign import a1_n3_campaign as journal
from test_scv_scalar import metadata as fragment_fixture, enumerated


def fixture(*,constant=False,identity=False):
    md,law,cat,kw=fragment_fixture()
    payload=json.loads(cat)
    # Positive a,b,c: -1/2, +1, -1; zero-grid row has +1/2.
    signs=(-1,1,-1,1);supports=((0,),(0,1),(1,2),(2,))
    if constant:signs=(1,1,1,1);supports=((0,),)*4
    if identity:supports=((),(0,1),(1,2),(2,))
    for row,s,p in zip(payload['components'],signs,supports):
        row.update(coefficient_sign=s,diagonal_pauli_support=p,coefficient_abs='UNUSED',rotation_angle='UNUSED')
    cat=json.dumps(payload).encode();kw=kw|dict(catalog_sha256=old.digest(cat))
    metadata=a.metadata_from_bytes(law,cat,system_qubits=3,**kw)
    return a.SignedControl(metadata),law,cat,kw


def scope(control,axis='X'):
    return cv.Scope('signed-toy-target',control.metadata.law_sha256,'toy-selected-full','toy-round',axis,1)


def toy_batch(control,cell,stage,n=12,*,axis='X',constant_feature=False):
    # Reuse old exact enumeration helper on INVENTED positive paths only.
    possibilities=[e for e,_ in enumerated(control.metadata) if control.metadata.cell(e)==cell]
    by_feature={control.value(e):e for e in possibilities}
    events=[by_feature[x] for x in sorted(by_feature)]
    token=n3.Domain('A1' if stage=='train' else 'B1-OL',axis,stage,cell).token
    records=[]
    for i in range(n):
        event=events[0 if constant_feature else i%len(events)]
        # Fixed toy observations; no conditional probability oracle or RNG.
        bit=((i//2)+(i%3==0))%2
        records.append(cv.Measurement(token,str(i),(event,),bit,F(1),'toy-path-'+str(i),'toy-selected-full'))
    return cv.Batch(scope(control,axis),stage,tuple(records))


def evaluation(control,counts=(4,10,4,4,4),axis='X'):
    return tuple((h,toy_batch(control,h,'prod',n,axis=axis)) for h,n in zip(old.CELLS,counts))


class SignedTests(unittest.TestCase):
    def setUp(self):self.control,self.law,self.cat,self.kw=fixture()

    def test_feature_negative_positive_half_values_and_identity(self):
        c=self.control
        self.assertEqual([r.value for r in c.metadata.components],[F(-1,2),F(1),F(-1),F(1,2)])
        identity,*_=fixture(identity=True)
        self.assertEqual(identity.value(n3.PathChoice(0,('a',)).event),0)
        self.assertEqual(c.value(n3.PathChoice(0,('b',)).event),1)
        self.assertEqual(c.value(n3.PathChoice(0,('c',)).event),-1)

    def test_rotation_last_not_product_and_explicit_wrong_rotation_rejected(self):
        e=n3.PathChoice(2,('b','c','a')).event
        self.assertEqual(self.control.value(e),F(-1,2))
        rte=NS(selected_component_ids=e.selected_component_ids,product_component_ids=('b','c'),rotation_component_id='a',taylor_order=2,phase=-1)
        self.assertEqual(self.control.value_from_rte(rte),F(-1,2))
        rte.rotation_component_id='b'
        with self.assertRaisesRegex(ValueError,'products then rotation'):self.control.value_from_rte(rte)

    def test_no_magnitude_angle_state_or_bits_in_feature(self):
        p=json.loads(self.cat)
        for r in p['components']:r['coefficient_abs']='NOT_A_NUMBER';r['rotation_angle']={'not':'feature'}
        b=json.dumps(p).encode();md=a.metadata_from_bytes(self.law,b,system_qubits=3,**(self.kw|dict(catalog_sha256=old.digest(b))))
        other=a.SignedControl(md)
        self.assertEqual(other.means,self.control.means)
        self.assertNotEqual(other.identity,self.control.identity) # full byte identity still tracked
        for e,_ in enumerated(md):self.assertEqual(other.value(e),self.control.value(e))

    def test_exact_means_second_moments_variances_match_all_toy_paths(self):
        c=self.control
        for cell,p in c.metadata.sampler.cells('B1-OL'):
            rows=[(e,w/p) for e,w in enumerated(c.metadata) if c.metadata.cell(e)==cell]
            mean=sum((w*c.value(e) for e,w in rows),F(0))
            second=sum((w*c.value(e)**2 for e,w in rows),F(0))
            self.assertEqual(c.means[int(cell)],mean)
            self.assertEqual(c.second_moments[int(cell)],second)
            self.assertEqual(c.variances[int(cell)],second-mean**2)
        self.assertEqual(c.means[0],F(-1,2));self.assertEqual(c.means[1],F(-1,3))
        self.assertEqual(c.means[2:],(F(-3,8),)*3)
        self.assertEqual(c.second_moments[0],F(1,4));self.assertEqual(c.second_moments[1],1)
        self.assertNotEqual(c.second_moments,c.means) # not an indicator

    def test_h0_constant_h1_varies_and_h234_free_rotation(self):
        c=self.control;values={h:set() for h in old.CELLS}
        for e,_ in enumerated(c.metadata):values[c.metadata.cell(e)].add(c.value(e))
        self.assertEqual(values['0'],{F(-1,2)});self.assertEqual(values['1'],{-1,1})
        for h in ('2','3','4'):self.assertEqual(values[h],{F(-1,2),-1,1})

    def test_moments_never_enumerate_event_paths(self):
        with patch.object(n3.ConditionalSampler,'draw',side_effect=AssertionError('no sampling')), \
             patch.object(a.Metadata,'rotation',side_effect=AssertionError('no path enumeration')):
            other=a.SignedControl(self.control.metadata)
        self.assertEqual(other.means,self.control.means)

    def test_zero_grid_label_is_kept_but_cannot_be_used_in_a_path(self):
        md=self.control.metadata
        self.assertIn('zero',[r.component_id for r in md.components]);self.assertEqual(md.sampler.law.labels.mass('zero'),0)
        for label in ('zero','absent'):
            with self.assertRaises(ValueError):self.control.value(n3.PathChoice(0,(label,)).event)
        p=json.loads(self.cat);p['components'][-1]['coefficient_sign']=-1
        b=json.dumps(p).encode();c=a.SignedControl(a.metadata_from_bytes(self.law,b,system_qubits=3,**(self.kw|dict(catalog_sha256=old.digest(b)))))
        self.assertEqual(c.means,self.control.means);self.assertEqual(c.second_moments,self.control.second_moments)

    def test_hash_H_preparation_and_label_alignment_reject_mixing(self):
        for changed in (dict(law_sha256='bad'),dict(catalog_sha256='bad'),dict(hamiltonian_hash='bad'),dict(preparation_hash='bad')):
            with self.assertRaises(ValueError):a.metadata_from_bytes(self.law,self.cat,system_qubits=3,**(self.kw|changed))
        md=self.control.metadata
        with self.assertRaises(ValueError):replace(md,components=md.components[::-1])
        with self.assertRaises(ValueError):replace(md,components=md.components[:-1])
        with self.assertRaises(ValueError):replace(md,components=(md.components[0],)*4)

    def test_bad_sign_support_and_qubit_domain_rejected(self):
        for sign in (0,2,True,1.,float('nan')):
            with self.assertRaises(ValueError):a.Component('x',sign,(0,))
        for support in ((1,0),(0,0),(0,1,2),(-1,),(True,),('0',)):
            with self.assertRaises(ValueError):a.Component('x',1,support)
        for n in (0,True,2):
            with self.assertRaises(ValueError):replace(self.control.metadata,system_qubits=n)

    def test_inputs_not_mutated_and_identity_mutation_detected(self):
        c=self.control; original=c.metadata.payload(); before=copy.deepcopy(original)
        source_support=[0]; component=a.Component('invented',-1,source_support)
        source_support.append(1)
        self.assertEqual(component.diagonal_pauli_support,(0,))
        a.SignedControl(c.metadata); c.value(n3.PathChoice(0,('a',)).event)
        self.assertEqual(c.metadata.payload(),before)
        self.assertEqual(c.metadata.components[0].diagonal_pauli_support,(0,))
        with self.assertRaises(FrozenInstanceError):c.metadata.components[0].coefficient_sign=1
        c.metadata.sampler.a='b'
        with self.assertRaisesRegex(ValueError,'identity changed'):c.value(n3.PathChoice(0,('a',)).event)

    def test_unbiased_conditional_centering_for_any_fixed_slope(self):
        c=self.control
        for cell,p in c.metadata.sampler.cells('B1-OL'):
            raw=res=F(0)
            for e,w in enumerated(c.metadata):
                if c.metadata.cell(e)!=cell:continue
                # An arbitrary analytic toy response for the identity check;
                # this table is NOT supplied to OLS (only toy bits are fitted).
                response=F(1,5) if e.selected_component_ids[-1]=='b' else F(-2,7)
                raw+=w/p*response;res+=w/p*(response-F(7,3)*(c.value(e)-c.means[int(cell)]))
            self.assertEqual(raw,res)
        self.assertNotEqual(c.means[1],c.means[2])

    def test_range_all_signs_of_coefficient_and_all_feature_endpoints(self):
        for lam,nu in product((F(-3),F(-1,2),F(0),F(5,3)),(F(-1),F(-1,3),F(1))):
            interval=a.residual_range(lam,nu)
            self.assertEqual(interval.hi-interval.lo,2+2*abs(lam))
            for b,u in product((-1,1),(F(-1),F(-1,2),F(0),F(1,2),F(1))):
                z=b-lam*(u-nu)
                self.assertLessEqual(interval.lo,z);self.assertLessEqual(z,interval.hi)

    def test_indicator_range_nonfinite_and_bad_mean_fail_closed(self):
        # Counterexample to reusing [0,1]: B=+1,u=-1,lambda=1,nu=-1/3
        # has residual 5/3; the copied indicator box would end at 2/3.
        correct=a.residual_range(1,F(-1,3))
        self.assertLessEqual(F(5,3),correct.hi)
        self.assertGreater(F(5,3),F(1)-F(1,3))
        for lo,hi in ((0,1),(-1,0),(1,-1),(-2,2)):
            with self.assertRaises(ValueError):a.residual_range(1,0,feature_lower=lo,feature_upper=hi)
        for lam,mean in ((float('nan'),0),(float('inf'),0),(1,2)):
            with self.assertRaises((ValueError,TypeError)):a.residual_range(lam,mean)
        with self.assertRaises(ValueError):a.SignedFrozenCV(old.ConditionalBank(old.select_fragment_control(self.control.metadata.base),'1'),toy_batch(self.control,'1','train'),(F(1),))

    def test_reused_OLS_rank_two_binary64_freeze_no_oracle_and_no_guard_adoption(self):
        train=toy_batch(self.control,'1','train');infos=[]
        with patch.object(cv,'fit_and_freeze',wraps=cv.fit_and_freeze) as fit:
            m=a.fit_h1(self.control,train,sink=infos.append)
        self.assertEqual(fit.call_count,1);self.assertEqual(infos[0]['rank'],2)
        slope=m.models[1].coefficient
        self.assertEqual(slope,F.from_float(float(slope)));self.assertEqual(m.training,train)
        self.assertEqual(infos[0]['range_guard'],'NOT_ADOPTED')
        self.assertEqual(tuple(x.coefficient for i,x in enumerate(m.models) if i!=1),(F(0),)*4)
        with self.assertRaises(FrozenInstanceError):m.models[1].frozen.coefficients=(F(0),)
        self.assertEqual(infos[0]['width'],2+2*abs(slope))

    def test_rank_deficiency_constant_catalog_and_constant_training_stop(self):
        for c in (fixture(constant=True)[0],self.control):
            train=toy_batch(c,'1','train',constant_feature=True);infos=[]
            with patch.object(cv,'fit_and_freeze',wraps=cv.fit_and_freeze) as fit:
                with self.assertRaisesRegex(ValueError,'rank deficient'):a.fit_h1(c,train,sink=infos.append)
            self.assertEqual(fit.call_count,1);self.assertEqual(infos[0]['rank'],1)

    def test_nonfinite_OLS_stops_no_clipping_or_refit(self):
        import numpy as np
        for bad in (float('inf'),float('nan')):
            with patch.object(np.linalg,'lstsq',return_value=(np.array([0.,bad]),None,None,None)) as solve:
                with self.assertRaisesRegex(ValueError,'nonfinite'):a.fit_h1(self.control,toy_batch(self.control,'1','train'))
            self.assertEqual(solve.call_count,1)

    def test_wrong_train_scope_domain_weight_cell_rejected(self):
        c=self.control;b=toy_batch(c,'1','train')
        bads=[replace(b,scope=replace(b.scope,law_id='other')),replace(b,stage='prod'),
              replace(b,records=tuple(replace(r,stream_id='old') for r in b.records)),
              replace(b,records=tuple(replace(r,weight=F(1,2)) for r in b.records)),toy_batch(c,'0','train')]
        for bad in bads:
            with self.assertRaises(ValueError):a.fit_h1(c,bad)

    def test_paired_means_variance_covariance_with_signed_values(self):
        c=self.control;m=a.fit_h1(c,toy_batch(c,'1','train'))
        # Keep the actual trained/frozen slope, never fit on evaluation.
        counts=(4,10,4,4,4);rows=evaluation(c,counts)
        with patch.object(cv,'fit_and_freeze',side_effect=AssertionError('no refit')):
            result=a.paired_evaluate(m,rows,counts=counts)
        p=dict(c.metadata.sampler.cells('B1-OL'));raw=res=vr=vz=cov=F(0)
        for h,b in rows:
            x=tuple(r.raw for r in b.records);z=m.apply(b);n=len(x)
            raw+=p[h]*sum(x,F(0))/n;res+=p[h]*sum(z,F(0))/n
            vr+=p[h]**2*variance(x)/n;vz+=p[h]**2*variance(z)/n
            mx=sum(x,F(0))/n;mz=sum(z,F(0))/n
            cov+=p[h]**2*sum(((v-mx)*(w-mz) for v,w in zip(x,z)),F(0))/(n-1)/n
            if h!='1':self.assertEqual(x,z)
        self.assertEqual(result['B1_OL']['mean'],raw);self.assertEqual(result['SCV']['mean'],res)
        self.assertEqual(result['B1_OL']['estimated_variance_of_mean'],vr)
        self.assertEqual(result['SCV']['estimated_variance_of_mean'],vz)
        self.assertEqual(result['estimated_covariance_of_means'],cov)
        self.assertEqual(result['estimated_variance_of_mean_difference'],vr+vz-2*cov)
        self.assertEqual(result['precision'],'NOT_CERTIFIED')
        self.assertEqual(sum(result['cells'][1]['feature_counts'].values()),10)
        self.assertIn('-1',result['cells'][1]['feature_counts'])

    def test_wrong_indicator_model_rejected_before_aggregation(self):
        c=self.control;m=a.fit_h1(c,toy_batch(c,'1','train'));active=m.models[1]
        wrong=replace(active,frozen=cv.FrozenCV(active.bank,m.training,active.frozen.coefficients))
        with self.assertRaisesRegex(ValueError,'indicator range forbidden'):
            replace(m,models=(m.models[0],wrong,*m.models[2:]))
        with self.assertRaisesRegex(ValueError,'signed model'):a.paired_evaluate(NS(),(),counts=(2,)*5)

    def test_fresh_stage_streams_full_cells_quota_and_scope_guards(self):
        c=self.control;m=a.fit_h1(c,toy_batch(c,'1','train'));rows=evaluation(c);counts=(4,10,4,4,4)
        with self.assertRaises(ValueError):a.paired_evaluate(m,rows[:-1],counts=counts)
        with self.assertRaises(ValueError):a.paired_evaluate(m,rows,counts=(4,8,4,4,4))
        b=rows[1][1]
        repeated=replace(b,records=tuple(replace(r,stream_id=m.training.records[0].stream_id) for r in b.records))
        with self.assertRaises(ValueError):a.paired_evaluate(m,(rows[0],('1',repeated),*rows[2:]),counts=counts)
        with self.assertRaises(ValueError):m.apply(replace(b,scope=replace(b.scope,axis='Y')))
        with self.assertRaises(ValueError):m.apply(replace(b,stage='cal'))

    def test_range_contains_actual_corrected_outputs_and_rejects_bad_enclosure(self):
        c=self.control;m=a.fit_h1(c,toy_batch(c,'1','train'));b=toy_batch(c,'1','prod')
        bounds=m.models[1].frozen.output_range
        for z in m.apply(b):self.assertTrue(bounds.lo<=z<=bounds.hi)
        with patch.object(a,'residual_range',return_value=a.Interval(F(0),F(0))):
            with self.assertRaisesRegex(ValueError,'outside signed range'):m.apply(b)

    def test_physical_price_once_train_is_not_free_cache_hits_still_charged(self):
        c=self.control;train=toy_batch(c,'1','train');rows=evaluation(c)
        def envelope(stage,rows):
            costs=tuple(cv.TrialCost(b.scope,r.stream_id,r.trial_id,stage,cv.FULL_SCOPE,
                (None,F(3),F(9),None,F(4)),F(0),F(0),123) for _,b in rows for r in b.records)
            return accounting.Envelope('invented-signed', 'invented-purpose',c.metadata.identity,c.identity,'X',stage,rows,costs,len(costs))
        t=envelope('train',(('1',train),));e=envelope('prod',rows)
        result=accounting.cost_summary((t,e));n=sum(len(b.records) for _,b in rows)
        self.assertEqual(result['physical_RZ'],9*(len(train.records)+n))
        self.assertEqual(result['train_RZ'],9*len(train.records));self.assertEqual(result['B1_OL_reference_evaluation_RZ'],9*n)
        self.assertEqual(result['cache_hits_returned'],len(train.records)+n)
        with self.assertRaisesRegex(ValueError,'unique full'):accounting.cost_summary((t,t,e))

    def test_existing_STOP_boundary_is_preserved_never_scientific_restart(self):
        with TemporaryDirectory() as d:
            out=Path(d);original='{"old_STOP": true}\n';(out/'STOP.json').write_text(original)
            with patch.object(n3,'load_certified_services',side_effect=AssertionError('no loader')), \
                 patch.object(n3,'pcg64_streams',side_effect=AssertionError('no RNG')):
                with self.assertRaisesRegex(ValueError,'STOP exists'):accounting.check_source(out,'toy',self.control)
            self.assertEqual((out/'STOP.json').read_text(),original)

    def test_no_real_loader_rng_engine_or_compiler_invoked(self):
        from trottertracks.algorithm_codesign import a1_engine_connection as engine
        with patch.object(n3,'load_certified_services',side_effect=AssertionError('real input')), \
             patch.object(n3,'pcg64_streams',side_effect=AssertionError('RNG')), \
             patch.object(n3.ConditionalSampler,'draw',side_effect=AssertionError('sampling')), \
             patch.object(engine,'apply_exact',side_effect=AssertionError('real state')), \
             patch.object(engine,'born_probability',side_effect=AssertionError('Born')), \
             patch.object(engine.SavedEngine,'selected_circuit',side_effect=AssertionError('real build')):
            c,*_=fixture();m=a.fit_h1(c,toy_batch(c,'1','train'))
            result=a.paired_evaluate(m,evaluation(c),counts=(4,10,4,4,4))
        self.assertEqual(result['range_guard'],'NOT_ADOPTED')


if __name__=='__main__':unittest.main()
