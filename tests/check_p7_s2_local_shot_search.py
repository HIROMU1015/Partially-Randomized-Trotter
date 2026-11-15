"""Artificial selection oracles plus metadata preparation; no real Q3."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_local_search_test',ROOT/'scripts/run_p7_s2_local_shot_search.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
FAIL='MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'


def toy():
    seed={k:'TOY' for k in ('configuration_id','total_view_id','binding_id','configuration','signal_packet','evaluation_target')}
    vector=[8]*13;vector[11:]=[5,4]
    seed['allocation']=dict(shots=[dict(m=m,N_X=n,N_Y=n) for m,n in enumerate(vector)])
    rows=[dict(m=m,axis=a,C=10+m,physical_id='TOY',original_candidate_id='TOY',total_candidate_id='TOY',
        price_class='SYNTHETIC_ONLY',price_source='TOY') for m in range(13) for a in ('X','Y')]
    seed['conditional_cost']=sum(vector[p['m']]*p['C'] for p in rows)
    single=[C.make_case(seed,rows,[8]*13,r,d) for r,d in ((12,1),(12,2),(11,1),(11,2))]
    transfer=[c for r in (12,11) for d in (1,2) for k in range(13) if k!=r for s in (1,2)
        if (c:=C.make_case(seed,rows,[8]*13,r,d,k,s)) is not None]
    return dict(seed=seed,single_cases=single,transfer_cases=transfer,maximum_scientific_calls=8,maximum_transfer_calls=4)


def answer(status=FAIL,upper=10,calls=1):
    return dict(status=status,MSE_lower=C.F.exact(upper),MSE_upper=C.F.exact(upper),scientific_evaluation_calls=calls)


class LocalSearchTests(unittest.TestCase):
    def test_success_first_has_no_speculative_calls(self):
        seen=[]
        def visit(c,selection):seen.append(c);return answer(C.PASS,1)
        x=C.search(toy(),visit)
        self.assertTrue(x['improvement_found']);self.assertEqual(x['new_scientific_calls'],1)
        self.assertEqual(len(seen),1);self.assertEqual(seen[0]['edit'],dict(r=12,d=1,k=None,s=0))

    def test_nonmonotone_second_not_pruned_and_seed_unchanged(self):
        inputs=toy();before=deepcopy(inputs);seen=[]
        def visit(c,selection):
            seen.append(c);return answer(FAIL if len(seen)==1 else C.PASS)
        x=C.search(inputs,visit)
        self.assertEqual(x['new_scientific_calls'],2)
        self.assertEqual(seen[1]['allocation']['shots'][12]['N_X'],2)
        self.assertEqual(seen[1]['allocation']['shots'][11]['N_X'],5)
        self.assertEqual(inputs,before)

    def test_eight_call_limit_and_no_changed_single_baselines(self):
        seen=[]
        x=C.search(toy(),lambda c,s:(seen.append(c) or answer()))
        self.assertEqual(x['new_scientific_calls'],8);self.assertEqual(x['transfer_scientific_calls'],4)
        self.assertEqual([(c['edit']['r'],c['edit']['d']) for c in seen[:4]],[(12,1),(12,2),(11,1),(11,2)])
        self.assertEqual(len({C.key(c) for c in seen}),8)
        for c in seen[:4]:
            changed=[r['m'] for r,b in zip(c['allocation']['shots'],toy()['seed']['allocation']['shots']) if r!=b]
            self.assertEqual(changed,[c['edit']['r']])

    def test_cache_success_needs_zero_new_calls(self):
        result=C.search(toy(),lambda c,s:answer(C.PASS,1,calls=0))
        self.assertTrue(result['improvement_found']);self.assertEqual(result['new_scientific_calls'],0)

    def test_inconclusive_continues_but_unknown_stops(self):
        seen=[]
        def visit(c,s):
            seen.append(c);return answer('INCONCLUSIVE' if len(seen)==1 else 'UNKNOWN')
        result=C.search(toy(),visit)
        self.assertFalse(result['improvement_found']);self.assertEqual(len(seen),2)
        self.assertEqual(result['stop_reason'],'UNKNOWN_GATE_OR_PERSISTENCE_STOP')

    def test_transfer_family_priority_uses_completed_risk(self):
        seen=[];selected=[]
        def visit(c,s):
            seen.append(c)
            if c['phase']=='TRANSFER':
                selected.append((c,s));return answer(C.PASS,1)
            return answer(upper=1 if c['family']==[11,2] else 10)
        C.search(toy(),visit)
        self.assertEqual(selected[0][0]['family'],[11,2])
        self.assertEqual(len(selected[0][1]['parent_results']),1)
        self.assertEqual(selected[0][1]['reason'],'FAMILY_MIN_VALID_UPPER_THEN_COST_MOVE_LEXICOGRAPHIC')

    def test_real_metadata_preparation_never_calls_risk(self):
        with patch.object(C.T,'evaluate',side_effect=AssertionError('No real risk')) as risk:
            inputs=C.build_inputs()
        self.assertEqual(risk.call_count,0)
        self.assertEqual(inputs['seed']['configuration_id'],'a6e557770a2e4d96585ad5ca5903c4626ae47dda9775e9a9babc756149c38d84')
        self.assertEqual([c['allocation']['shots'][12]['N_X'] for c in inputs['single_cases']],[3,2,4,4])
        self.assertEqual([c['allocation']['shots'][11]['N_X'] for c in inputs['single_cases']],[13,13,12,11])
        self.assertTrue(all(c['conditional_cost']<435481090 for c in inputs['single_cases']+inputs['transfer_cases']))

    def test_saved_seed_reuse_checks_target_law_and_identity(self):
        inputs=C.build_inputs()
        with patch.object(C.T,'evaluate',side_effect=AssertionError('No re-evaluation')):
            saved=C.cached_record(inputs['seed'],inputs)
            self.assertEqual(saved['status'],C.PASS);self.assertEqual(saved['scientific_evaluation_calls'],0)
            bad=deepcopy(inputs['seed']);bad['configuration_id']='FOREIGN'
            with self.assertRaises(ValueError):C.cached_record(bad,inputs)
            bad=deepcopy(inputs);bad['options']['probability_rule']='FOREIGN_LAW'
            with self.assertRaises(ValueError):C.cached_record(inputs['seed'],bad)


if __name__=='__main__':unittest.main(verbosity=2)
