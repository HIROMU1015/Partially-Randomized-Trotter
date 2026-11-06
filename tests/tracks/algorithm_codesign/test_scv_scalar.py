"""Invented dyadic laws and fixed outcome tables; no real data or RNG."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
import json
import unittest
from unittest.mock import patch

from trottertracks.algorithm_codesign import a1_control_variates as cv
from trottertracks.algorithm_codesign import a1_n3_connection as n3
from trottertracks.algorithm_codesign import scv_scalar as a


def metadata(masses=(F(1,4),F(1,4),F(1,2),F(0)),fragments=('f0','f1','f2','unused')):
    record=dict(uniform_grid_size=2**53,application_label_order='products_then_rotation',orders=[0,2],
        order_masses=[dict(exact='3/4'),dict(exact='1/4')],
        label_masses=[dict(label=l,exact=str(p)) for l,p in zip(('a','b','c','zero'),masses)])
    catalog=dict(hamiltonian_hash='toy-H',preparation_hash='toy-preparation',
        components=[dict(component_id=l,df_fragment_id=f) for l,f in zip(('a','b','c','zero'),fragments)])
    law=json.dumps(record).encode();cat=json.dumps(catalog).encode()
    kw=dict(law_sha256=a.digest(law),catalog_sha256=a.digest(cat),
        hamiltonian_hash='toy-H',preparation_hash='toy-preparation',a='a',b='b')
    return a.metadata_from_bytes(law,cat,**kw),law,cat,kw


def enumerated(md):
    law=md.sampler.law
    for n,pn in zip(law.orders.labels,law.orders.masses):
        for labels in product(('a','b','c'),repeat=n+1):
            event=n3.PathChoice(n,labels).event
            p=pn
            for label in labels:p*=law.labels.mass(label)
            if p:yield event,p


def scope(md,axis='X'):
    return cv.Scope('toy-target',md.law_sha256,'toy-selected-full','m0',axis,1)


def batch(control,cell,stage,n=8,axis='X',constant_feature=False,independent_response=False):
    md=control.metadata
    events=[event for event,_ in enumerated(md) if md.cell(event)==cell]
    # Deterministic cover both feature values, where they exist.
    by_value={control.value(e):e for e in events}
    events=[by_value[k] for k in sorted(by_value)]
    method='A1' if stage=='train' else 'B1-OL'
    token=n3.Domain(method,axis,stage,cell).token
    rows=[]
    for i in range(n):
        event=events[0 if constant_feature else i%len(events)]
        bit=(i//2)%2 if independent_response else 1-control.value(event)
        rows.append(cv.Measurement(token,str(i),(event,),bit,F(1),'toy-path-'+str(i),'toy-selected-full'))
    return cv.Batch(scope(md,axis),stage,tuple(rows))


class ScalarTests(unittest.TestCase):
    def setUp(self):self.md,*_=metadata();self.control=a.select_fragment_control(self.md)

    def test_exact_means_and_second_moments_all_five_cells(self):
        for cell,p in self.md.sampler.cells('B1-OL'):
            records=[(e,w/p) for e,w in enumerated(self.md) if self.md.cell(e)==cell]
            mean=sum((w*self.control.value(e) for e,w in records),F(0))
            second=sum((w*self.control.value(e)**2 for e,w in records),F(0))
            self.assertEqual(mean,self.control.means[int(cell)])
            self.assertEqual(second,self.control.second_moments[int(cell)])
            self.assertEqual(second-mean**2,self.control.variances[int(cell)])

    def test_h0_constant_h1_nonconstant_and_range(self):
        values={h:set() for h in a.CELLS}
        for e,_ in enumerated(self.md):values[self.md.cell(e)].add(self.control.value(e))
        self.assertEqual(len(values['0']),1);self.assertEqual(values['1'],{0,1})
        self.assertTrue(all(v<=set((0,1)) for v in values.values()))

    def test_correct_conditional_centering_and_unbiased_any_fixed_slope(self):
        slope=F(7,5)
        for cell,p in self.md.sampler.cells('B1-OL'):
            raw=residual=F(0)
            for e,w in enumerated(self.md):
                if self.md.cell(e)!=cell:continue
                # Artificial conditional bit mean, not a Born computation.
                response=F(1,3) if self.control.value(e) else F(-2,5)
                raw+=w/p*response
                residual+=w/p*(response-slope*(self.control.value(e)-self.control.means[int(cell)]))
            self.assertEqual(raw,residual)
        self.assertNotEqual(self.control.means[0],self.control.means[1])

    def test_selection_deterministic_metadata_only_tie_and_zero_mass_fragment(self):
        # h1 f1=1/3, f2=2/3 tie; increasing mask chooses f1.
        self.assertEqual(self.control.fragments,('f1',));self.assertEqual(self.control.means[1],F(1,3))
        self.assertNotIn('unused',self.md.positive_fragments)
        self.assertEqual(a.select_fragment_control(self.md).identity,self.control.identity)
        self.assertEqual(tuple(__import__('inspect').signature(a.select_fragment_control).parameters),('metadata',))

    def test_no_eligible_group_stops(self):
        md,*_=metadata((F(1,4),F(1,4),F(1,2),F(0)),('only','only','only','unused'))
        with self.assertRaisesRegex(ValueError,'no eligible'):a.select_fragment_control(md)

    def test_balance_guard_rejects_tiny_groups(self):
        md,*_=metadata((F(1,4),F(1,64),F(47,64),F(0)),('only','rare','only','unused'))
        with self.assertRaisesRegex(ValueError,'no eligible'):a.select_fragment_control(md)

    def test_bad_group_empty_full_noncanonical_unknown(self):
        for group in ((),self.md.positive_fragments,('f2','f1'),('absent',)):
            with self.assertRaises(ValueError):a.FragmentControl(self.md,group)

    def test_byte_and_saved_identity_guards(self):
        md,law,cat,kw=metadata()
        for changed in (dict(law_sha256='bad'),dict(catalog_sha256='bad'),dict(hamiltonian_hash='bad'),dict(preparation_hash='bad')):
            with self.assertRaises(ValueError):a.metadata_from_bytes(law,cat,**(kw|changed))
        with self.assertRaises(ValueError):replace(md,catalog=md.catalog[:-1])
        with self.assertRaises(ValueError):replace(md,catalog=md.catalog+(md.catalog[0],))

    def test_law_mutation_detected_and_invalid_event_rejected(self):
        for e in (n3.PathChoice(0,('zero',)).event,n3.PathChoice(0,('unknown',)).event):
            with self.assertRaises(ValueError):self.control.value(e)
        self.md.sampler.a='b'
        with self.assertRaisesRegex(ValueError,'identity changed'):self.control.value(n3.PathChoice(0,('b',)).event)

    def test_fit_rank2_freeze_and_inactive_coefficients(self):
        training=batch(self.control,'1','train',32,independent_response=True)
        models=a.fit_cells(self.control,training)
        self.assertEqual(tuple(m.coefficient for i,m in enumerate(models.models) if i!=1),(F(0),)*4)
        for h in a.CELLS:
            b=batch(self.control,h,'prod')
            if h!='1':self.assertEqual(models.apply(b),tuple(r.raw for r in b.records))
        self.assertEqual(models.training,training)
        with self.assertRaises(Exception):models.models[1].frozen.coefficients=(F(0),)

    def test_rank_deficiency_stops_no_refit(self):
        with patch.object(cv,'fit_and_freeze',wraps=cv.fit_and_freeze) as fit:
            with self.assertRaisesRegex(ValueError,'rank2'):a.fit_cells(self.control,batch(self.control,'1','train',constant_feature=True))
            self.assertEqual(fit.call_count,1)

    def test_nonfinite_slope_stops_and_range_guard_preserves_failed_fit(self):
        import numpy as np
        training=batch(self.control,'1','train')
        with patch.object(np.linalg,'lstsq',return_value=(np.array([0.,float('nan')]),None,None,None)):
            with self.assertRaisesRegex(ValueError,'nonfinite'):a.fit_cells(self.control,training)
        infos=[]
        with self.assertRaisesRegex(ValueError,'range guard'):
            a.fit_cells(self.control,training,range_guard=3,sink=infos.append)
        self.assertEqual(infos[0]['rank'],2);self.assertGreater(infos[0]['width'],3)

    def test_frozen_range_contains_all_bit_feature_combinations(self):
        models=a.fit_cells(self.control,batch(self.control,'1','train',independent_response=True))
        f=models.models[1].frozen
        for bit,u in product((0,1),repeat=2):
            z=1-2*bit-f.coefficients[0]*(u-self.control.means[1])
            self.assertLessEqual(f.output_range.lo,z);self.assertLessEqual(z,f.output_range.hi)
        self.assertEqual(f.output_range.hi-f.output_range.lo,2+abs(f.coefficients[0]))

    def test_paired_weighted_means_variances_and_no_refit(self):
        models=a.fit_cells(self.control,batch(self.control,'1','train',independent_response=True))
        rows=tuple((h,batch(self.control,h,'prod',n)) for h,n in zip(a.CELLS,(4,64,4,4,4)))
        with patch.object(cv,'fit_and_freeze',side_effect=AssertionError('refit forbidden')):
            r=a.paired_evaluate(models,rows)
        p=dict(self.md.sampler.cells('B1-OL'))
        raw=sum((p[h]*sum((v.raw for v in b.records),F(0))/len(b.records) for h,b in rows),F(0))
        scv=sum((p[h]*sum(models.apply(b),F(0))/len(b.records) for h,b in rows),F(0))
        self.assertEqual(r['B1_OL']['mean'],raw);self.assertEqual(r['SCV']['mean'],scv)
        self.assertFalse(r['independent_method_campaigns'])
        self.assertGreaterEqual(r['estimated_variance_of_mean_difference'],0)
        self.assertEqual(r['precision'],'NOT_CERTIFIED')

    def test_nonzero_paired_correction_and_covariance_formula(self):
        from statistics import variance
        models=a.fit_cells(self.control,batch(self.control,'1','train',independent_response=True))
        # Fix an exact toy slope to test nontrivial paired arithmetic, not OLS accuracy.
        active=replace(models.models[1],frozen=cv.FrozenCV(models.models[1].bank,models.training,(F(1),)))
        models=replace(models,models=(models.models[0],active,*models.models[2:]))
        rows=tuple((h,batch(self.control,h,'prod',n)) for h,n in zip(a.CELLS,(4,64,4,4,4)))
        r=a.paired_evaluate(models,rows)
        b=rows[1][1];differences=tuple(z-v.raw for z,v in zip(models.apply(b),b.records))
        p=dict(self.md.sampler.cells('B1-OL'))['1']
        self.assertEqual(r['estimated_variance_of_mean_difference'],p*p*variance(differences)/64)
        self.assertLess(r['SCV']['estimated_variance_of_mean'],r['B1_OL']['estimated_variance_of_mean'])
        self.assertNotEqual(r['SCV']['mean'],r['B1_OL']['mean'])

    def test_training_reuse_wrong_law_missing_cell_and_scope_stops(self):
        models=a.fit_cells(self.control,batch(self.control,'1','train',independent_response=True))
        rows=tuple((h,batch(self.control,h,'prod',n)) for h,n in zip(a.CELLS,(4,64,4,4,4)))
        with self.assertRaises(ValueError):a.paired_evaluate(models,rows[:-1])
        b=rows[1][1]
        with self.assertRaises(ValueError):models.apply(replace(b,scope=replace(b.scope,law_id='other')))
        reused=replace(b,records=tuple(replace(r,stream_id=models.training.records[0].stream_id) for r in b.records))
        with self.assertRaises(ValueError):a.paired_evaluate(models,(rows[0],('1',reused),*rows[2:]))
        wrong_cell=replace(rows[0][1],records=b.records[:4])
        with self.assertRaises(ValueError):a.paired_evaluate(models,(('0',wrong_cell),*rows[1:]))

    def test_old_three_controls_are_constant_within_every_cell(self):
        bank=self.md.sampler.bank
        for cell,_ in self.md.sampler.cells('B1-OL'):
            values={bank.values((e,)) for e,_ in enumerated(self.md) if self.md.cell(e)==cell}
            self.assertEqual(len(values),1)
            # Every centered old control is identically zero conditionally.
            nu=next(iter(values))
            self.assertEqual({tuple(x-y for x,y in zip(v,nu)) for v in values},{(0,0,0)})


if __name__=='__main__':unittest.main()
