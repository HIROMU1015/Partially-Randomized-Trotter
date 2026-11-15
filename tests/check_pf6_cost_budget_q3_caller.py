"""Toy allocations and actual worker/monitor path; no real H4 Q3."""
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
sys.path.insert(0,str(ROOT/'tests'))
from check_pf_shortlist_q3_caller import artificial, MOCK
spec=importlib.util.spec_from_file_location('_pf6_budget_test',ROOT/'scripts/run_pf6_cost_budget_q3_fixed.py')
F=importlib.util.module_from_spec(spec);sys.modules[spec.name]=F;spec.loader.exec_module(F)


class BudgetCallerTests(unittest.TestCase):
    def test_exact_breakpoint_maximum_and_shape(self):
        g=F.generate([3,2],[5,11],Fraction(23),'A_SHAPE')
        self.assertEqual(g['shots'],[2,1])
        self.assertEqual(g['conditional_cost'],21)
        self.assertEqual(g['unused_budget'],F.exact(Fraction(2)))
        self.assertGreater(g['next_breakpoint_cost'],23)

    def test_price_rule_and_equal_vector_can_be_deduplicated(self):
        a=F.generate([3,2],[5,11],Fraction(30),'A_SHAPE')
        b=F.generate([3,2],[5,11],Fraction(30),'B_PRICE')
        self.assertEqual(a['shots'],[2,1]);self.assertEqual(b['shots'],[3,1])
        aa=F.generate([3,2],[5,11],Fraction(23),'A_SHAPE')
        bb=F.generate([3,2],[5,11],Fraction(23),'B_PRICE')
        self.assertEqual(aa['shots'],bb['shots'])
        self.assertEqual(F.generate([3,2],[5,11],Fraction(37),'A_SHAPE')['shots'],[3,2])

    def test_missing_noninteger_nonpositive_or_infeasible_prices(self):
        self.assertIsNone(F.generate([3,2],[5,11],15,'A_SHAPE'))
        for prices in ([5], [5,0], [5,None], [5,1.5], [5,True]):
            with self.assertRaises((ValueError,TypeError)):
                F.generate([3,2],prices,100,'B_PRICE')
        with self.assertRaises(ValueError):F.generate([3,2],[5,11],100,'FOREIGN_RULE')

    def test_huge_prices_and_XY_count_only_bounds(self):
        big=10**80
        g=F.generate([3,2],[5*big,11*big],23*big,'A_SHAPE')
        self.assertEqual(g['shots'],[2,1]);self.assertEqual(g['conditional_cost'],21*big)
        b=F.count_bounds([dict(m=0,N_X=2,N_Y=3),dict(m=1,N_X=1,N_Y=2)])
        self.assertEqual(b['maximum_round_outcomes'],12)
        self.assertEqual(b['verification_pairs_upper'],72)
        self.assertEqual(b['transition_pairs_upper'],84)

    def test_actual_installed_worker_saves_API_and_restores_hooks(self):
        case,inputs,binding,view=artificial();before=deepcopy((case,inputs,view))
        old=F.M.worker
        with tempfile.TemporaryDirectory() as folder,patch.object(F,'metadata',return_value=(binding,view)),patch.object(F,'check_source'):
            root=Path(folder)
            with F.installed_workers(root):receipt=F.M.worker(case,inputs,MOCK,root)
            self.assertIs(F.M.worker,old)
            result=json.loads((root/'RESULT.json').read_text())
            self.assertEqual(json.loads((root/'API_RETURN.json').read_text()),[result['record']])
            self.assertEqual(receipt['scientific_evaluation_calls'],1)
            for name in ['RESOURCE_FINAL.json','PARTIAL_COARSE.json','PARTIAL_LEMMA.json',
                         'PARTIAL_TOTAL_DP.json','PERSISTED.json','SEGMENTS.jsonl']:
                self.assertTrue((root/name).exists(),name)
            for k in ('sufficient_N','global_RMSE_bound','G','G_qualified','winner'):
                self.assertIsNone(result[k])
            self.assertEqual(result['MC_calls'],0)
        self.assertEqual((case,inputs,view),before)

    def test_actual_supervisor_monitor_failure_preserves_started_worker(self):
        case,inputs,binding,view=artificial();inputs['concurrent_configurations']=1
        ctx=F.M.mp.get_context('fork');ready=ctx.Event();release=ctx.Event()
        original=F.T.evaluate
        def blocked(*args,**kwargs):
            ready.set()
            if not release.wait(10):raise AssertionError('Toy worker was not released')
            return original(*args,**kwargs)
        def fail_monitor(*args):
            if not ready.wait(10):raise AssertionError('Toy worker did not start')
            release.set();raise RuntimeError('ARTIFICIAL_MONITOR_FAILURE')
        with tempfile.TemporaryDirectory() as folder,patch.object(F,'metadata',return_value=(binding,view)),\
                patch.object(F,'check_source'),patch.object(F.T,'evaluate',side_effect=blocked):
            root=Path(folder);(root/case['label']).mkdir()
            with F.installed_workers(root):
                result=F.M.supervise([case],inputs,MOCK,root,monitor=fail_monitor,poll_seconds=.01)
            self.assertFalse(result['errors']);self.assertEqual(len(result['monitor_errors']),1)
            self.assertEqual(result['receipts'][0]['scientific_evaluation_calls'],1)
            self.assertTrue(result['receipts'][0]['api_return_saved'])
            for name in ['API_RETURN.json','RESULT.json','RESOURCE_FINAL.json','PERSISTED.json']:
                self.assertTrue((root/case['label']/name).exists())
            exits=[c for p in root.glob('PROCESS_EXITS_*.jsonl') for line in p.read_text().splitlines()
                   for c in json.loads(line)['children']]
            self.assertTrue(exits);self.assertTrue(all(c['exitcode']==0 for c in exits))


if __name__=='__main__':unittest.main(verbosity=2)
