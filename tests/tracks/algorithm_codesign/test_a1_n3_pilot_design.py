"""Invented masses, fixed bit lists and toy-law metadata; no real inputs/RNG."""
from fractions import Fraction as F
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from trottertracks.algorithm_codesign import a1_n3_pilot_design as p
from trottertracks.algorithm_codesign import a1_control_variates as cv


def toy():
    law=cv.OccurrenceLaw(cv.CategoricalLaw((0,2),(F(3,4),F(1,4)),'toy'),
        cv.CategoricalLaw(('a','b'),(F(1,4),F(3,4)),'toy'))
    sampler=p.n3.ConditionalSampler(law,'toy','a','b')
    scope=cv.Scope('toy-target','toy','toy-wrapper','m0','X',F(1))
    return sampler,scope


def batch(sampler,scope,key,bits):
    stream=p.n3.Domain('B1-OL',scope.axis,'prod',key).token
    # A fixed invented word (not a PRNG) chooses a representative toy path.
    path=sampler.draw('B1-OL',key,lambda:0)
    records=tuple(cv.Measurement(stream,str(i),(path.event,),bit,F(1),'toy-path','toy-wrapper') for i,bit in enumerate(bits))
    return cv.Batch(scope,'prod',records)


class PilotDesignTests(unittest.TestCase):
    def test_finite_scope_752(self):
        self.assertEqual(p.BUDGET.counts,dict(A1_train=160,B1_OL_probe=80,A1_evaluation=256,B1_OL_evaluation=256))
        self.assertEqual(p.BUDGET.total,752)

    def test_all_positive_rare_strata_keep_floor(self):
        tiny=F(1,10**80);masses=(1-4*tiny,tiny,tiny,tiny,tiny)
        n=p.allocate_evaluation(masses,(0,)*5,(1,)*5)
        self.assertEqual(sum(n),128);self.assertEqual(n[1:],(4,4,4,4))

    def test_non_normalized_zero_or_float_mass_rejected(self):
        for masses in ((F(1,5),)*4+(F(0),),(F(1,4),)*5):
            with self.assertRaises(ValueError):p.allocate_evaluation(masses,(0,)*5,(1,)*5)
        with self.assertRaises(TypeError):p.allocate_evaluation((.2,)*5,(0,)*5,(1,)*5)

    def test_variance_and_price_both_affect_allocation(self):
        masses=(F(1,5),)*5
        n=p.allocate_evaluation(masses,(1,0,0,0,0),(1,1,1,1,1))
        self.assertGreater(n[0],n[1])
        costly=p.allocate_evaluation(masses,(1,)*5,(100,1,1,1,1))
        self.assertLess(costly[0],costly[1]);self.assertGreaterEqual(costly[0],4)

    def test_zero_variance_and_zero_price_are_not_zero_quota(self):
        n=p.allocate_evaluation((F(1,5),)*5,(0,)*5,(0,)*5)
        self.assertEqual(n,(26,26,26,25,25))

    def test_negative_or_infeasible_inputs_rejected(self):
        for values,costs in (((-1,0,0,0,0),(1,)*5),((0,)*5,(-1,1,1,1,1))):
            with self.assertRaises(ValueError):p.allocate_evaluation((F(1,5),)*5,values,costs)
        with self.assertRaises(ValueError):p.allocate_evaluation((F(1,5),)*5,(0,)*5,(1,)*5,total=19)

    def test_five_cell_mean_keeps_duplicate_control_patterns_separate(self):
        sampler,scope=toy();cells=sampler.cells('B1-OL');rows=[];expected=F(0)
        patterns=[]
        for i,(key,mass) in enumerate(cells):
            bits=(i%2,)*4;rows.append((key,batch(sampler,scope,key,bits)))
            expected+=mass*(1-2*bits[0]);patterns.append(sampler.bank.values(rows[-1][1].records[0].events))
        self.assertLess(len(set(patterns)),5)
        self.assertEqual(p.pilot_mean(sampler,'B1-OL',scope,rows,(4,)*5),expected)

    def test_missing_cell_wrong_quota_and_stream_overlap_stop(self):
        sampler,scope=toy();rows=[(key,batch(sampler,scope,key,(0,1,0,1))) for key,_ in sampler.cells('B1-OL')]
        with self.assertRaises(ValueError):p.pilot_mean(sampler,'B1-OL',scope,rows[:-1],(4,)*5)
        with self.assertRaises(ValueError):p.pilot_mean(sampler,'B1-OL',scope,rows,(5,4,4,4,4))
        antecedent=cv.Batch(scope,'train',rows[0][1].records)
        with self.assertRaisesRegex(ValueError,'disjoint'):p.pilot_mean(sampler,'B1-OL',scope,rows,(4,)*5,antecedents=(antecedent,))

    def test_target_mismatch_and_extra_method_stop(self):
        sampler,scope=toy();rows=[(key,batch(sampler,scope,key,(0,1))) for key,_ in sampler.cells('B1-OL')]
        wrong=cv.Scope('different','toy','toy-wrapper','m0','X',F(1))
        with self.assertRaises(ValueError):p.pilot_mean(sampler,'B1-OL',wrong,rows,(2,)*5)
        with self.assertRaises(ValueError):p.pilot_mean(sampler,'B0',scope,rows,(2,)*5)

    def test_A1_reuses_frozen_correction_on_independent_unconditional_evaluation(self):
        sampler,scope=toy();training=[]
        for key,_ in sampler.cells('B1-OL'):
            path=sampler.draw('B1-OL',key,lambda:0)
            stream=p.n3.Domain('A1','X','train',key).token
            training.append(cv.Batch(scope,'train',tuple(cv.Measurement(stream,str(i),(path.event,),i%2,F(1),'toy','toy-wrapper') for i in range(4))))
        model,rank=p.n3.freeze(sampler.bank,p.n3.merge_training(training),4)
        self.assertEqual(rank,4)
        path=sampler.draw('B0','all',lambda:0);stream=p.n3.Domain('A1','X','prod','all').token
        records=tuple(cv.Measurement(stream,str(i),(path.event,),i%2,F(1),'toy','toy-wrapper') for i in range(8))
        evaluation=cv.Batch(scope,'prod',records)
        expected=sum(model.apply(evaluation),F(0))/8
        self.assertEqual(p.pilot_mean(sampler,'A1',scope,(('all',evaluation),),(8,),model=model),expected)
        overlapping=cv.FrozenCV(sampler.bank,cv.Batch(scope,'train',records),model.coefficients)
        with self.assertRaisesRegex(ValueError,'disjoint'):
            p.pilot_mean(sampler,'A1',scope,(('all',evaluation),),(8,),model=overlapping)

    def test_helpers_never_bind_real_engine_or_draw_rng(self):
        with patch.object(p.n3,'load_certified_services',side_effect=AssertionError('no real bundle')),patch.object(p.n3,'pcg64_streams',side_effect=AssertionError('no RNG')):
            p.allocate_evaluation((F(1,5),)*5,(0,)*5,(1,)*5)
            sampler,scope=toy();rows=[(key,batch(sampler,scope,key,(0,1))) for key,_ in sampler.cells('B1-OL')]
            p.pilot_mean(sampler,'B1-OL',scope,rows,(2,)*5)


if __name__=='__main__':unittest.main()
