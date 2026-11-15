"""Synthetic conditional minimum selection; no model or price evaluation."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_pf6_budget_summary_test',ROOT/'scripts/summarize_pf6_cost_budget_q3_fixed.py')
S=importlib.util.module_from_spec(spec);sys.modules[spec.name]=S;spec.loader.exec_module(S)


def row(label,cost,status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD',shots=3):
    return dict(label=label,base_label=label,rule='ARTIFICIAL_ONLY',method='FULL_PF',
        status=status,conditional_cost=cost,configuration_id=label,result_source={'fixture':'ARTIFICIAL'},
        allocation=dict(allocation_id=label,shots=[dict(m=0,N_X=shots,N_Y=shots)]))


class SummaryTests(unittest.TestCase):
    def test_cost_not_shot_total_and_model_fail_excluded(self):
        result=S.confirmed_minima([row('FEWER_SHOTS',100,shots=1),row('CHEAPER',90,shots=4),
            row('FAIL',1,'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD')])['FULL_PF']
        self.assertEqual(result['conditional_cost'],90)
        self.assertEqual(result['allocations'][0]['label'],'CHEAPER')
        self.assertIsNone(result['sufficient_N']);self.assertIsNone(result['G_qualified'])

    def test_unresolved_remains_and_ties_are_not_discarded(self):
        result=S.confirmed_minima([row('A',90),row('B',90),row('U',1,'UNKNOWN'),
            row('I',2,'INCONCLUSIVE')])['FULL_PF']
        self.assertEqual([v['label'] for v in result['allocations']],['A','B'])
        self.assertEqual(result['unresolved_allocations'],['U','I'])
        self.assertIsNone(S.confirmed_minima([])['FULL_PF']['conditional_cost'])


if __name__=='__main__':unittest.main(verbosity=2)
