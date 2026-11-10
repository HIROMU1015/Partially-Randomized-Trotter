"""m7 input delta and q8/R16 artificial regression; no molecular arrays."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests'),str(ROOT/'src')]
import run_h6_df19_rpe_m7 as m7
import run_h6_df19_rpe_m6 as m6
from check_h6_df19_rpe_d2 import toy,candidate,ready,old


def plan():return json.loads((ROOT/m7.PLAN_PATH).read_bytes())


class M7Tests(unittest.TestCase):
    def test_new_scope_keeps_old_D2_guard_and_fixed_two_candidates(self):
        p=plan();m7.validate_m7(p)
        self.assertEqual(len(p['rows']),2)
        with self.assertRaises(ValueError):m7.backend.validate_plan(p)
        with self.assertRaises(ValueError):m6.validate_m6(p)
        for mutate in (
                lambda x:x['rows'][0].update(m=6),
                lambda x:x['rows'][1]['candidate'].update(q=128),
                lambda x:x['rows'][1]['candidate'].update(K=4),
                lambda x:x['fixed_PR_allocation'].__setitem__(0,2),
                lambda x:x['configuration_reuse'].update(alpha_axis=.025),
                lambda x:x['scientific_binding'].update(state='foreign')):
            bad=deepcopy(p);mutate(bad)
            with self.assertRaises(ValueError):m7.validate_m7(bad)

    def test_q8_R16_matches_existing_independent_spectral_mean(self):
        ctx=toy();c=candidate(T=1.4,q=8);c['R_outer']=16
        ref=ready(ctx,c);actual=m7.backend.evolve(ctx,c,ref)
        with patch.object(old,'weights',m7.backend.schedule.weights):
            expected=old.compare_prefix(ctx.state,[old.spectrum(a) for a in ctx.matrices[:2]],
                old.spectrum(ctx.tails[1]['matrix']),.21,.5,c['formula'],8,T=1.4,
                total_steps=16,cutoffs=(2,))['finite_rows'][0]
        for key,want in [('corrected',expected['corrected_signal']),('raw',expected['raw_signal'])]:
            z=actual['signal'][key]
            self.assertLess(abs(complex(z['real'],z['imag'])-want),2e-13)
        self.assertAlmostEqual(actual['signal']['log_B'],expected['log_B'],places=14)
        self.assertEqual(actual['event_count'],128)
        self.assertEqual(actual['allocation'],plan()['fixed_PR_allocation'])
        self.assertLess(actual['signal']['allowance_evidence']['raw_corrected_vector_closure'],1e-12)

    def test_m7_scalar_binding_and_actual_unknown_without_price_lookup(self):
        p=plan()
        for row in p['rows']:
            sig=dict(reference=dict(real=1.,imag=0.),corrected=dict(real=.999,imag=.0001),
                raw=dict(real=.999,imag=.0001),total_signed_bias=dict(real=-.001,imag=.0001),
                log_B=0.,allowance_evidence=dict(maximum_intermediate_norm=1.01,
                    raw_corrected_vector_closure=1e-14))
            record=dict(candidate=row['candidate'],signal=sig,origin='SYNTHETIC_ONLY',
                signed_weights=[1.],allocation=[16],event_count=128)
            out=m7.analyze_m7(record,row,p,state_energy=0.,state_residual=1e-12,
                reference_evidence=dict(signal_difference=2e-12))
            self.assertEqual((out['m'],out['q_RPE'],out['q_PF']),(7,128,8))
            self.assertEqual(out['stage'],'D2_M7')
            self.assertEqual(out['ideal_numerical_pilot']['status'],'CONDITIONAL_ROUND_PASS')
            self.assertIsNone(out['actual_implementation']['N_X'])
            self.assertIsNone(out['compiled_price']);self.assertIsNone(out['G'])
            self.assertEqual(out['price_coverage'],'NOT_EVALUATED_SEPARATE_PRICE_TASK')
            with self.assertRaises(ValueError):m7.analyze_m7(record,row,p,
                state_energy=0.,state_residual=0.,reference_evidence={})


if __name__=='__main__':unittest.main()
