"""Full PF/S2 m12=1 fixed caller: scope, prices, entry and decision. No real H4 Q3."""
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
spec=importlib.util.spec_from_file_location('_full_pf_m12_one_test',ROOT/'scripts/run_full_pf_s2_m12_one_fixed.py')
X=importlib.util.module_from_spec(spec);sys.modules[spec.name]=X;spec.loader.exec_module(X)


def guarded_inputs():
    with patch.object(X.T,'evaluate',side_effect=AssertionError('No H4 risk')),\
            patch.object(X.C,'search',side_effect=AssertionError('No search')):
        return X.build_inputs()


class FixedFullPFOneShot(unittest.TestCase):
    def test_fixed_case_prices_scope_and_unmodified_basis(self):
        old=X.F.document(X.BASE,X.OLD_INPUT);before=deepcopy(old)
        inputs=guarded_inputs();self.assertEqual(old,before)
        self.assertEqual(len(inputs['cases']),1);self.assertEqual(inputs['maximum_scientific_calls'],1)
        case=inputs['cases'][0];two=old['cases'][0]
        self.assertEqual([r['N_X'] for r in case['allocation']['shots']],X.VECTOR)
        self.assertEqual([r['N_Y'] for r in case['allocation']['shots']],X.VECTOR)
        self.assertEqual(case['allocation']['shots'][:12],two['allocation']['shots'][:12])
        self.assertEqual([(t['m'],t['axis'],t['C'],t['price_source']) for t in case['price_terms']],
                         [(t['m'],t['axis'],t['C'],t['price_source']) for t in two['price_terms']])
        self.assertEqual(case['conditional_cost'],531533828);self.assertEqual(case['reduction_RZ'],27923558)
        self.assertEqual(inputs['previous_achieved_cost_RZ'],559457386)
        for key in ('statistical_contract','options','adopted_evaluation','kernel_source','acceleration'):
            self.assertEqual(inputs[key],old[key])
        self.assertIsNone(X.C.cached_record(case,inputs))
        self.assertEqual(X.C.cached_record(two,inputs)['status'],X.MEETS)
        self.assertEqual([r['m12'] for r in inputs['comparisons']],[4,2])

    def test_foreign_configuration_and_law_rejected(self):
        inputs=guarded_inputs();two=deepcopy(inputs['saved_cache'][-1]['case'])
        foreign=deepcopy(two);foreign['configuration_id']='FOREIGN'
        with self.assertRaises(ValueError):X.C.cached_record(foreign,inputs)
        bad=deepcopy(inputs);bad['options']['probability_rule']='FOREIGN'
        with self.assertRaises(ValueError):X.C.cached_record(two,bad)

    def test_actual_execute_uses_only_one_case_and_new_output(self):
        inputs=dict(cases=[dict(label='TOY_FULL_ONE')],maximum_scientific_calls=1)
        def supervised(cases,x,source,run):
            self.assertEqual(cases,inputs['cases']);self.assertEqual(source['caller'],X.CALLER)
            (run/'PROCESS_EXITS_toy.jsonl').write_text(json.dumps(dict(children=[dict(exitcode=0)]))+'\n')
            return dict(receipts=[],errors=[],monitor_errors=[])
        fake=dict(status=X.MEETS,scientific_evaluation_calls=1,MSE_lower=0,MSE_upper=0,result_source='SYNTHETIC_ONLY')
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
            self.assertEqual(selection['reason'],'SEPARATELY_APPROVED_FULL_PF_S2_FIXED_M12_ONE_ONLY')
            self.assertFalse((root/X.OLD_DIRECTORY/'run01').exists())

    def test_summary_decision_for_meets_exceeds_and_held(self):
        inputs=guarded_inputs();tau=inputs['adopted_evaluation']['MSE_upper'];real=X.T._selected
        twice=dict(numerator=tau['numerator']*2,denominator=tau['denominator'])
        for status,lower,upper,achieved,one,cheapest in (
                (X.MEETS,dict(numerator=1,denominator=10**7),dict(numerator=1,denominator=10**6),531533828,'MODEL_FEASIBLE',None),
                (X.EXCEEDS,twice,twice,559457386,'MODEL_INFEASIBLE',2),
                ('INCONCLUSIVE',dict(numerator=1,denominator=10**7),twice,559457386,'HELD_INCONCLUSIVE',None),
                ('UNKNOWN',None,None,559457386,'HELD_UNKNOWN',None)):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);p=root/X.DIRECTORY;(p/'run01').mkdir(parents=True)
                (p/'INPUTS.json').write_text(json.dumps(inputs))
                (p/'run01/OUTCOME.json').write_text(json.dumps(dict(status=status,scientific_evaluation_calls=1,
                    MSE_lower=lower,MSE_upper=upper,result_source='SYNTHETIC_ONLY')))
                (p/'run01/EXECUTION.json').write_text(json.dumps(dict(source=dict(commit='TOY'))))
                (p/'PROCESS_RETURN.json').write_text(json.dumps(dict(returncode=0)))
                with patch.object(X,'ROOT',root),patch.object(X.T,'_selected',side_effect=lambda _r,s:real(ROOT,s)),\
                        patch('builtins.print'):
                    X.summarize()
                join=json.loads((p/'C_RESULT_JOIN.json').read_text())
                self.assertEqual(join['achieved_conditional_cost_RZ'],achieved);self.assertEqual(join['one_shot'],one)
                self.assertEqual(join['achieved_cost_updated'],status==X.MEETS)
                self.assertEqual(join['conditional_cost_RZ'],531533828)
                self.assertEqual(None if join['cheapest_in_scope'] is None else join['cheapest_in_scope']['cheapest_model_feasible_m12'],cheapest)
                self.assertEqual([r['m12'] for r in join['reference_results']],[4,2])
                exact=json.loads((p/'EXACT_COMPARISON.json').read_text())
                self.assertEqual([r['m12'] for r in exact['rows']],[4,2,1])
                self.assertTrue(all(r['status']==X.MEETS for r in exact['rows'][:2]))


if __name__=='__main__':unittest.main(verbosity=2)
