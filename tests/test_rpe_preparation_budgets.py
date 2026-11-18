"""A resumed ordinary entry retains its initial scientific preparation caps."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_rpe_schedule_search as S
import run_rpe_method_optimizer as M


class PreparationBudgets(unittest.TestCase):
    def test_width_fixed_scope_rejects_resume_budget_increase(self):
        with tempfile.TemporaryDirectory()as temp:
            driver=object.__new__(S.WidthDriver);driver.run=Path(temp)
            driver.problem={};driver.widths={};driver.methods=[];driver.state={'families':{}}
            driver.live_ready_refill=False
            driver.spec={'executor':{'distributed':{'preparation':{'budgets':{'signal':1}}}}}
            driver.check_fixed_inputs();before=(driver.run/'FIXED_INPUTS.json').read_bytes()
            driver.check_fixed_inputs()
            driver.spec['executor']['distributed']['preparation']['budgets']['signal']=2
            with self.assertRaisesRegex(ValueError,'FIXED_INPUTS_CHANGED'):driver.check_fixed_inputs()
            self.assertEqual(before,(driver.run/'FIXED_INPUTS.json').read_bytes())
            driver.spec['executor']['distributed']['preparation']['budgets']['signal']=1
            driver.spec['pf_completion']={'max_candidates':2}
            with self.assertRaisesRegex(ValueError,'FIXED_INPUTS_CHANGED'):driver.check_fixed_inputs()

    def test_standard_round_resolves_stage_caps_separately_from_q3(self):
        self.assertEqual(M.preparation_budgets({'max_new_q3_per_method':{'PR_DF':'UNLIMITED'}}),
            {'signal':0,'compile':0,'audit':0})
        spec={'executor':{'distributed':{'preparation':{'budgets':{'compile':2}}}}}
        self.assertEqual(M.preparation_budgets(spec),{'signal':0,'compile':2,'audit':0})


if __name__=='__main__':unittest.main()
