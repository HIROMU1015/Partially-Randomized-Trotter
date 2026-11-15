"""Exact proposal joins and finite sequential control; artificial risk only."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
spec=importlib.util.spec_from_file_location('_ld3_budget_test',ROOT/'scripts/run_pauli_ld3_pf_budget_q3_fixed.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
from check_pf_shortlist_q3_caller import artificial,MOCK
from check_rpe_pauli_integration import forbid_science


class BudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with forbid_science(): cls.inputs=C.build_inputs()

    def test_fixed_four_order_exact_costs_caps_and_unchanged_risk(self):
        x=self.inputs;prior=C.document(dict(commit=C.BASE,path=C.L.INPUT))
        self.assertEqual([c['label'] for c in x['cases']],['P7__A_SHAPE','P7__B_PRICE','FULL_PF__A_SHAPE','FULL_PF__B_PRICE'])
        self.assertEqual([C.rational(c['conditional_cost']) for c in x['cases']],
            [Fraction(30009021696,77),Fraction(456463851968,1155),Fraction(10729281172,21),Fraction(625936421828,1155)])
        for c in x['cases']:
            self.assertEqual(len(c['price_terms']),26)
            self.assertEqual(C.rational(c['conditional_cost'])+C.rational(c['unused_budget']),c['budget'])
            self.assertEqual(C.rational(c['conditional_cost']),sum(C.rational(t['N_times_C_ref']) for t in c['price_terms']))
            self.assertTrue(all(1<=r['N_X']<=C.L.VECTOR[r['m']] and r['N_X']==r['N_Y'] for r in c['allocation']['shots']))
            self.assertIsNone(C.cached_record(c,x))
        for k in ('options','statistical_contract','adopted_evaluation','acceleration'):
            self.assertEqual(x[k],prior[k])
        reused=C.cached_record(x['seed_case'],x)
        self.assertEqual(reused['scientific_evaluation_calls'],0);self.assertEqual(reused['status'],C.PASS)
        self.assertFalse(x['price_quality']['qualified_for_candidate'])

    def test_rational_breakpoints_match_finite_independent_chain(self):
        caps=[3,4];prices=[Fraction(7,3),Fraction(11,5)];budget=Fraction(14)
        for rule in C.RULES:
            weights=[Fraction(n) if rule=='A_SHAPE' else Fraction(n)/c for n,c in zip(caps,prices)]
            points=sorted({Fraction(k)/w for n,w in zip(caps,weights) for k in range(1,n+1)})
            possible=[]
            for point in points:
                shots=[max(1,min(n,(point*w).numerator//(point*w).denominator)) for n,w in zip(caps,weights)]
                cost=sum(n*c for n,c in zip(shots,prices))
                if cost<=budget:possible.append((point,shots,cost))
            g=C.generate(caps,prices,budget,rule)
            self.assertEqual((g['lambda_value'],g['shots'],g['conditional_cost']),possible[-1])

    def test_price_physical_law_source_mismatch_rejected(self):
        with forbid_science():
            b,v=C.L.metadata(self.inputs['seed_case']);table=C.document(C.PRICE)
            for k,value in [('original_candidate_id','OTHER'),('C_ref_exact','1/2'),('binding_id','OTHER')]:
                bad=deepcopy(table);bad['rows'][0][k]=value
                with self.assertRaises(ValueError):C.joined_prices(bad,b,v)
            bad=deepcopy(table);bad['qualified_for_candidate']=True
            with self.assertRaises(ValueError):C.joined_prices(bad,b,v)

    def test_sequential_first_pass_and_cheaper_inconclusive(self):
        calls=[]
        def visit(case,sequence):
            calls.append(case['label'])
            return dict(status='INCONCLUSIVE' if sequence==1 else C.PASS,scientific_evaluation_calls=1)
        result=C.search(self.inputs,visit)
        self.assertEqual(calls,self.inputs['evaluation_order'][:2]);self.assertEqual(result['new_scientific_calls'],2)
        self.assertFalse(result['minimal_within_generated_set']);self.assertEqual(result['cheaper_unresolved'],calls[:1])
        result=C.search(self.inputs,lambda c,s:dict(status=C.FAIL,scientific_evaluation_calls=1))
        self.assertEqual(result['new_scientific_calls'],4);self.assertEqual(result['achieved_conditional_cost'],self.inputs['seed_cost'])
        result=C.search(self.inputs,lambda c,s:dict(status='UNKNOWN',scientific_evaluation_calls=1))
        self.assertEqual(result['new_scientific_calls'],1);self.assertEqual(len(result['unevaluated']),3)

    def test_duplicate_vectors_merge_and_changed_cached_contract_refused(self):
        def identical(caps,prices,budget,rule):
            cost=sum(prices)
            return dict(shots=[1]*len(caps),rule=rule,conditional_cost=cost,unused_budget=Fraction(budget)-cost)
        with forbid_science(),patch.object(C,'generate',side_effect=identical):
            merged=C.build_inputs()
        self.assertEqual(len(merged['cases']),1);self.assertEqual(len(merged['cases'][0]['aliases']),4)
        bad=deepcopy(self.inputs);bad['options']['probability_rule']='OTHER_LAW'
        with self.assertRaisesRegex(ValueError,'contract differs'):C.cached_record(bad['seed_case'],bad)

    def test_actual_entry_artificial_worker_persistence_hooks_and_stop(self):
        case,x,b,v=artificial();x.update(cases=[case],maximum_scientific_calls=1,concurrent_configurations=1,seed_cost=999,saved_cache=[])
        case['conditional_cost']=1
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/C.DIRECTORY).mkdir(parents=True);(root/C.INPUT).write_text(json.dumps(x))
            (root/C.CALLER).parent.mkdir();(root/C.CALLER).write_text('ARTIFICIAL_ONLY')
            with patch.object(C,'ROOT',root),patch.object(C,'check_source'),patch.object(C,'build_inputs',return_value=x),\
                    patch.object(C.subprocess,'check_output',return_value='ARTIFICIAL_SOURCE'),\
                    patch.object(C.R,'blob',side_effect=lambda sha,path:(root/path).read_bytes()),\
                    patch.object(C.L,'metadata',return_value=(b,v)),patch.object(C.L,'check_source'),\
                    patch.dict(C.validate_result.__globals__,ROOT=root):
                C.execute()
            step=root/C.DIRECTORY/'run01/01';folder=step/case['label']
            for f in ('API_RETURN.json','RESULT.json','RESOURCE_FINAL.json','PERSISTED.json','SEGMENTS.jsonl'):
                self.assertTrue((folder/f).exists(),f)
            h=json.loads((root/C.DIRECTORY/'run01/SEARCH_HISTORY.json').read_text())
            self.assertEqual(h['new_scientific_calls'],1);self.assertEqual(h['best_confirmed'],case['label'])
            self.assertIsNone(h['G_qualified']);self.assertTrue(h['mandatory_stop'])


if __name__=='__main__':unittest.main(verbosity=2)
