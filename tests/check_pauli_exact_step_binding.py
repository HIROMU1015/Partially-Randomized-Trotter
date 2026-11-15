"""Saved metadata and artificial mutations only; no signal/compile/fit/Q3."""
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trottertracks import rpe_optimization_pipeline as I
from trace_pauli_step_binding import trace, PACKETS, read

ROOT=Path(__file__).resolve().parents[1]
B=I.providers(str(ROOT)).B
HELPER=I._pauli_helpers(str(ROOT))
BAPI=SimpleNamespace(validate=B._validate,exact_scalar=B.a.exact_scalar)
REF=dict(commit=I.A_PAULI_SOURCE,
    path='docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json')


def fixture(width=Fraction(1,5),M=12):
    p=read(REF)['packets'][0];spec=deepcopy(p['configuration']['spec'])
    spec['M']=M;spec['delta_pf']=B.a.exact_value(width)
    config=B.generate(spec)
    sources={str(r['m']):r['mean_record']['source'] for r in p['rounds'][:M+1]}
    return config,sources


def context(config,sources,m):
    return HELPER.job(config,m,sources[str(m)],BAPI)['physical_context']


class ExactStep(unittest.TestCase):
    def test_saved_round_trace_reproduces_A_decisions_without_scientific_calls(self):
        with patch.object(I,'produce_signals',side_effect=AssertionError('No signal')), \
                patch.object(I,'compare',side_effect=AssertionError('No statistics/cost')):
            result=trace()
        self.assertEqual(len(result['rows']),52)
        self.assertEqual(result['new_science_calls'],dict(signal=0,sampling=0,compile=0,fit=0,Q3=0))
        for prefix in PACKETS:
            rows=[r for r in result['rows'] if r['prefix']==prefix and r['case']=='exact_1_5']
            self.assertEqual([r['m'] for r in rows if r['matched']],[0,1])
            self.assertEqual([r['m'] for r in rows if r['A_support']['decision']=='UNSUPPORTED'],list(range(2,13)))
            self.assertTrue(all(r['same_S2_lowering_operands'] for r in rows))
            self.assertTrue(all(r['price'] is None and not r['price_record_accepted'] for r in rows))

    def test_exact_and_binary64_widths_change_integer_q_at_m1(self):
        exact,sources=fixture();rounded,_=fixture(Fraction(.2))
        self.assertEqual([r['q_PF'] for r in exact['rounds']], [1,2,3,5,9,17,33,65,129,257,513,1025,2049])
        self.assertEqual([r['q_PF'] for r in rounded['rounds']], [1,1,2,4,8,16,32,64,128,256,512,1024,2048])
        wrong=context(rounded,sources,1)
        check=I.pauli_round_binding(B,exact,exact['rounds'][1],wrong)
        self.assertFalse(check['matched']);self.assertIn('q',check['differences'])
        self.assertFalse(check['same_S2_lowering_operands'])

    def test_representation_only_exact_values_pass_without_tolerance(self):
        config,sources=fixture(M=1)
        for row in config['rounds']:
            ctx=context(config,sources,row['m']);before=deepcopy(ctx)
            self.assertTrue(I.pauli_round_binding(B,config,row,ctx)['matched'])
            c=ctx['physical_candidate']
            c['T']=B.a.exact_value(Fraction(c['T']));c['delta']=B.a.exact_value(Fraction(c['delta']))
            self.assertTrue(I.pauli_round_binding(B,config,row,ctx)['matched'])
            c['delta']=math.nextafter(float(B.a.exact_scalar(c['delta'])),math.inf)
            self.assertFalse(I.pauli_round_binding(B,config,row,ctx)['matched'])
            self.assertNotEqual(ctx,before)

    def test_rounding_agreement_does_not_bypass_exact_provider_contract(self):
        config,sources=fixture();row=config['rounds'][2];ctx=context(config,sources,2)
        check=I.pauli_round_binding(B,config,row,ctx)
        self.assertEqual(check['differences'],['delta'])
        self.assertTrue(check['same_S2_lowering_operands'])
        self.assertFalse(check['binary64_step_is_canonical']);self.assertFalse(check['matched'])
        with self.assertRaisesRegex(ValueError,'m=2 q=3 fields=delta'):
            I._pauli_plan(ROOT,I.providers(str(ROOT)),config,None,sources)

    def test_actual_planner_passes_safe_rounds_and_retains_source_and_id(self):
        config,sources=fixture(M=1)
        target=dict(quantity='ORIGINAL_DF_H_TARGET_ENERGY',binding=config['spec']['binding'],
            energy_shift_ha=0,source=REF)
        request=I.plan(ROOT,config,target=target,signal_source=REF,implementation_sources=sources)
        I._validate(ROOT,request)
        for row,job in zip(config['rounds'],request['pending']['jobs']):
            self.assertEqual(job['physical_context']['physical_candidate']['id'],row['candidate']['id'])
            self.assertEqual(job['physical_context']['numerical_recipe']['source'],sources[str(row['m'])])
            self.assertEqual(job['physical_context']['structure'],row['structure'])
        self.assertEqual(request['science_calls'],0);self.assertIsNone(request['C'])


if __name__=='__main__':unittest.main(verbosity=2)
