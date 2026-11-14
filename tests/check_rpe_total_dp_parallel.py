"""Artificial exact-state tests only. No saved signal or real risk input."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from check_rpe_total_decoder_risk import make,CASES
from trottertracks import rpe_total_dp_parallel as P

R=P.R


def tables(args):
    entry=R.K.prepare(args['config'],args['signal_packet'],args['allocations'],
        evaluation_target=args['evaluation_target'],statistical_contract=args['statistical_contract'],
        integration=args['integration'])[0]
    return R.tables_for(entry,args['options'])


class ParallelDPTests(unittest.TestCase):
    def test_deferred_exact_sums_with_signed_coefficients_and_cache_invalidation(self):
        values={'value':Fraction(0)};expected=Fraction(0)
        for n,d,c in ((2**5000+1,2**5100,Fraction(-7,3)),
                      (2**4999-3,2**5100,Fraction(11,5)),
                      (7,13,Fraction(2**2000+1,2**2100)),(0,17,Fraction(9,11))):
            mass=P.Mass(n,d);P._accumulate(values,'value',mass,c);expected+=Fraction(n,d)*c
            self.assertEqual(P._res_value(values,'value'),expected)

    def test_full_original_DP_output_exactly_matches_all_artificial_cases(self):
        engine=P.Accelerator(workers=1,outcome_batch=3)
        for name in CASES:
            args=make(name);tt=tables(args);scope=args['options']['total_state_refinement']
            before=deepcopy(tt)
            old=R.total_state_dp(tt,h=10.,target=args['evaluation_target']['energy_ha'],scope=scope)
            new=engine(tt,h=10.,target=args['evaluation_target']['energy_ha'],scope=scope)
            self.assertEqual(new,old,name);self.assertEqual(tt,before)
            self.assertTrue(R.K.p.same(new,old),name)
        args=make(vector=[1]*13,raws=[(.25,-.5)]*13,partial=None,total=None)
        tt=tables(args);scope=dict(source={'fixture':'ARTIFICIAL_13_ROUNDS'},
            maximum_frontier_states=8,maximum_transition_pairs=5000)
        self.assertTrue(R.K.p.same(engine(tt,h=10.,target=0.,scope=scope),
                                   R.total_state_dp(tt,h=10.,target=0.,scope=scope)))

    def test_process_counts_and_task_partition_leave_outputs_identical(self):
        args=make('zero_tie_rich');tt=tables(args);scope=args['options']['total_state_refinement']
        expected=R.total_state_dp(tt,h=10.,target=0.,scope=scope)
        for workers,batch in ((2,1),(2,4),(4,2)):
            engine=P.Accelerator(workers=workers,outcome_batch=batch)
            self.assertEqual(engine(tt,h=10.,target=0.,scope=scope),expected)
            self.assertTrue(any(r['workers']>1 for r in engine.observations))

    def test_scope_boundary_uses_exact_same_state_prefix_and_residual(self):
        args=make('four_rounds');tt=tables(args)
        for states,pairs in ((0,0),(0,100000),(1,17),(2,51),(5,1000),(100,3)):
            scope=dict(source={'fixture':'ARTIFICIAL'},maximum_frontier_states=states,maximum_transition_pairs=pairs)
            expected=R.total_state_dp(tt,h=10.,target=0.,scope=scope)
            for workers in (1,2):
                engine=P.Accelerator(workers=workers,outcome_batch=2)
                self.assertEqual(engine(tt,h=10.,target=0.,scope=scope),expected,(states,pairs,workers))

    def test_big_integer_mass_and_signed_zero_are_not_float_aggregated(self):
        args=make('mixed_parity');tt=tables(args)
        for table in tt:
            factor=2**3000+7
            table['denominator']*=factor
            for outcome in table['outcomes']:outcome['weight']*=factor
        scope=args['options']['total_state_refinement']
        expected=R.total_state_dp(tt,h=10.,target=0.,scope=scope)
        self.assertEqual(P.Accelerator(workers=2)(tt,h=10.,target=0.,scope=scope),expected)
        self.assertNotEqual(P._phase_hex(0),P._phase_hex(1<<63))
        self.assertEqual(P._phase_hex(1<<63),'-0x0.0p+0')

    def test_invalid_support_unknowns_and_numerical_failure_remain_rejected(self):
        args=make('four_rounds');tt=tables(args);scope=args['options']['total_state_refinement']
        for mutation in ('mass','phase','q'):
            bad=deepcopy(tt)
            if mutation=='mass':bad[0]['denominator']+=1
            if mutation=='phase':bad[0]['outcomes'][0]['principal']=float('nan')
            if mutation=='q':bad[1]['q']=3
            self.assertEqual(P.Accelerator(workers=1)(bad,h=10.,target=0.,scope=scope),
                             R.total_state_dp(bad,h=10.,target=0.,scope=scope))
        with patch.object(R.T,'step',return_value=(__import__('numpy').array([float('nan')]),None)):
            self.assertEqual(P.Accelerator(workers=1)(tt,h=10.,target=0.,scope=scope)['reason'],
                             'NONFINITE_TOTAL_DECODER_TRANSITION')

    def test_opt_in_reuses_existing_evaluate_and_restores_original_kernel(self):
        args=make('zero_tie_rich');original=R.total_state_dp;expected=R.evaluate(**args)
        with P.installed(workers=2,outcome_batch=2) as engine:
            self.assertEqual(R.evaluate(**args),expected)
            self.assertTrue(engine.observations)
            with self.assertRaisesRegex(ValueError,'Nested/concurrent'):
                P.Accelerator(workers=1)
        self.assertIs(R.total_state_dp,original)
        for workers,batch in ((0,1),(True,1),(1,0)):
            with self.assertRaises(ValueError):P.Accelerator(workers=workers,outcome_batch=batch)


if __name__=='__main__':unittest.main(verbosity=2)
