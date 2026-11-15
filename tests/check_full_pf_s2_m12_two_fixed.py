"""Full PF/S2 m12=2 fixed caller: scope, prices, entry and summary. No real H4 Q3."""
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_full_pf_m12_two_test',ROOT/'scripts/run_full_pf_s2_m12_two_fixed.py')
X=importlib.util.module_from_spec(spec);sys.modules[spec.name]=X;spec.loader.exec_module(X)


def guarded_inputs():
    with patch.object(X.T,'evaluate',side_effect=AssertionError('No H4 risk')),\
            patch.object(X.C,'search',side_effect=AssertionError('No search')):
        return X.build_inputs()


class FixedFullPFTwoShot(unittest.TestCase):
    def test_fixed_case_prices_scope_and_unmodified_seed(self):
        old=X.F.document(X.BASE,X.OLD+'INPUTS.json');before=deepcopy(old)
        inputs=guarded_inputs();self.assertEqual(old,before)
        self.assertEqual(len(inputs['cases']),1);self.assertEqual(inputs['maximum_scientific_calls'],1)
        case=inputs['cases'][0];seed=next(c for c in old['cases'] if c['label']=='FULL_PF_S2_16h_over_7__B_PRICE')
        self.assertEqual(case['configuration_id'],X.CONFIGURATION)
        self.assertEqual([r['N_X'] for r in case['allocation']['shots']],X.VECTOR)
        self.assertEqual([r['N_Y'] for r in case['allocation']['shots']],X.VECTOR)
        self.assertEqual(case['allocation']['shots'][:12],seed['allocation']['shots'][:12])
        self.assertEqual(len(case['price_terms']),26)
        self.assertEqual([(t['m'],t['axis'],t['C']) for t in case['price_terms']],[(t['m'],t['axis'],t['C']) for t in seed['price_terms']])
        self.assertEqual(case['conditional_cost'],559457386);self.assertEqual(case['reduction_RZ'],55847116)
        self.assertEqual(case['conditional_cost_m0_m11']+case['conditional_cost_m12'],case['conditional_cost'])
        self.assertEqual(inputs['previous_achieved_cost_RZ'],615304502)
        for key in ('statistical_contract','options','adopted_evaluation'):self.assertEqual(inputs[key],old[key])
        self.assertIsNone(X.C.cached_record(case,inputs))
        self.assertEqual(X.C.cached_record(seed,inputs)['status'],X.C.PASS)
        self.assertEqual([r['m12'] for r in inputs['comparisons']],[4])

    def test_foreign_configuration_law_and_price_rejected(self):
        inputs=guarded_inputs();seed=deepcopy(inputs['saved_cache'][1]['case'])
        foreign=deepcopy(seed);foreign['configuration_id']='FOREIGN'
        with self.assertRaises(ValueError):X.C.cached_record(foreign,inputs)
        bad=deepcopy(inputs);bad['options']['probability_rule']='FOREIGN'
        with self.assertRaises(ValueError):X.C.cached_record(seed,bad)
        wrong=deepcopy(seed);wrong['price_terms'][24]['C']+=1
        with self.assertRaises(ValueError):X.price_rows(wrong)

    def test_actual_execute_uses_only_one_case_and_new_output(self):
        inputs=dict(cases=[dict(label='TOY_FULL_TWO')],maximum_scientific_calls=1)
        def supervised(cases,x,source,run):
            self.assertEqual(cases,inputs['cases']);self.assertEqual(source['caller'],X.CALLER)
            (run/'PROCESS_EXITS_toy.jsonl').write_text(json.dumps(dict(children=[dict(exitcode=0)]))+'\n')
            return dict(receipts=[],errors=[],monitor_errors=[])
        fake=dict(status=X.C.PASS,scientific_evaluation_calls=1,MSE_lower=0,MSE_upper=0,result_source='SYNTHETIC_ONLY')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/X.DIRECTORY).mkdir(parents=True)
            (root/X.INPUT).write_text(json.dumps(inputs));(root/X.CALLER).parent.mkdir(parents=True);(root/X.CALLER).write_text('TOY')
            with patch.object(X,'ROOT',root),patch.object(X,'check_source'),patch.object(X,'build_inputs',return_value=inputs),\
                    patch.object(X.subprocess,'check_output',return_value='TOY_SOURCE'),\
                    patch.object(X.R,'blob',side_effect=lambda sha,path:(root/path).read_bytes()),\
                    patch.object(X.C,'cached_record',return_value=None),patch.object(X.F,'installed_workers',return_value=nullcontext()),\
                    patch.object(X.M,'supervise',side_effect=supervised) as call,patch.object(X.C,'validate_result',return_value=fake):
                X.execute()
            self.assertEqual(call.call_count,1)
            selection=json.loads((root/X.DIRECTORY/'run01/SELECTION.json').read_text())
            self.assertEqual(selection['reason'],'SEPARATELY_APPROVED_FULL_PF_S2_FIXED_M12_TWO_ONLY')
            self.assertEqual(json.loads((root/X.DIRECTORY/'run01/OUTCOME.json').read_text())['status'],X.C.PASS)

    def test_summary_updates_cost_only_on_pass(self):
        inputs=guarded_inputs();tau=inputs['adopted_evaluation']['MSE_upper'];real=X.T._selected
        for status,upper,achieved in ((X.C.PASS,dict(numerator=1,denominator=10**6),559457386),
                ('MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD',dict(numerator=tau['numerator']*2,denominator=tau['denominator']),615304502),
                ('MODEL_BOUND_INCONCLUSIVE',tau,615304502)):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);p=root/X.DIRECTORY;(p/'run01').mkdir(parents=True)
                (p/'INPUTS.json').write_text(json.dumps(inputs))
                (p/'run01/OUTCOME.json').write_text(json.dumps(dict(status=status,scientific_evaluation_calls=1,
                    MSE_lower=dict(numerator=1,denominator=10**7),MSE_upper=upper,result_source='SYNTHETIC_ONLY')))
                (p/'run01/EXECUTION.json').write_text(json.dumps(dict(source=dict(commit='TOY'))))
                (p/'PROCESS_RETURN.json').write_text(json.dumps(dict(returncode=0)))
                with patch.object(X,'ROOT',root),patch.object(X.T,'_selected',side_effect=lambda _r,s:real(ROOT,s)),\
                        patch('builtins.print'):
                    X.summarize()
                join=json.loads((p/'C_RESULT_JOIN.json').read_text())
                self.assertEqual(join['achieved_conditional_cost_RZ'],achieved)
                self.assertEqual(join['achieved_cost_updated'],status==X.C.PASS)
                self.assertEqual(join['conditional_cost_RZ'],559457386)
                exact=json.loads((p/'EXACT_COMPARISON.json').read_text())
                self.assertEqual([r['m12'] for r in exact['rows']],[4,2])
                self.assertEqual(exact['rows'][0]['status'],X.C.PASS)


if __name__=='__main__':unittest.main(verbosity=2)
