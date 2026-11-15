"""Check the actual reused entry's scope and fixed 1-shot binding; no H4 Q3."""
from copy import deepcopy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_one_shot_test',ROOT/'scripts/run_p7_s2_m12_one_fixed.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)


class OneShotTests(unittest.TestCase):
    def test_fixed_one_scope_and_unmodified_old_rows(self):
        old=C.F.document(C.BASE,C.OLD_INPUT);before=deepcopy(old)
        with patch.object(C.T,'evaluate',side_effect=AssertionError('No H4 risk')),\
                patch.object(C.C,'search',side_effect=AssertionError('No search')):
            inputs=C.build_inputs()
        self.assertEqual(len(inputs['cases']),1);self.assertEqual(inputs['maximum_scientific_calls'],1)
        case=inputs['cases'][0]
        self.assertEqual([r['N_X'] for r in case['allocation']['shots']],C.VECTOR)
        self.assertEqual([r['N_Y'] for r in case['allocation']['shots']],C.VECTOR)
        self.assertEqual(case['conditional_cost'],376203472)
        self.assertEqual(inputs['previous_achieved_cost_RZ'],395962678)
        self.assertEqual(case['allocation']['shots'][:-1],old['cases'][0]['allocation']['shots'][:-1])
        self.assertEqual(old,before)
        self.assertIsNone(C.C.cached_record(case,inputs))
        self.assertEqual([r['m12'] for r in inputs['comparisons']],[4,3,2])

    def test_actual_execute_uses_only_one_case_and_new_output(self):
        inputs=dict(cases=[dict(label='TOY_ONE')],maximum_scientific_calls=1)
        def supervised(cases,x,source,run):
            self.assertEqual(cases,inputs['cases']);self.assertEqual(source['caller'],C.CALLER)
            (run/'PROCESS_EXITS_toy.jsonl').write_text(json.dumps(dict(children=[dict(exitcode=0)]))+'\n')
            return dict(receipts=[],errors=[],monitor_errors=[])
        fake=dict(status=C.C.PASS,scientific_evaluation_calls=1,MSE_lower=0,MSE_upper=0,result_source='SYNTHETIC_ONLY')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/C.DIRECTORY).mkdir(parents=True)
            (root/C.INPUT).write_text(json.dumps(inputs));(root/C.CALLER).parent.mkdir(parents=True);(root/C.CALLER).write_text('TOY')
            with patch.object(C,'ROOT',root),patch.object(C,'check_source'),patch.object(C,'build_inputs',return_value=inputs),\
                    patch.object(C.subprocess,'check_output',return_value='TOY_SOURCE'),\
                    patch.object(C.R,'blob',side_effect=lambda sha,path:(root/path).read_bytes()),\
                    patch.object(C.C,'cached_record',return_value=None),patch.object(C.F,'installed_workers',return_value=nullcontext()),\
                    patch.object(C.Q.M,'supervise',side_effect=supervised) as call,patch.object(C.C,'validate_result',return_value=fake):
                C.execute()
            self.assertEqual(call.call_count,1)
            selection=json.loads((root/C.DIRECTORY/'run01/SELECTION.json').read_text())
            self.assertEqual(selection['reason'],'SEPARATELY_APPROVED_FIXED_M12_ONE_ONLY')
            self.assertEqual(selection['case']['label'],'TOY_ONE')
            self.assertFalse((root/C.OLD_DIRECTORY/'run01').exists())


if __name__=='__main__':unittest.main(verbosity=2)
