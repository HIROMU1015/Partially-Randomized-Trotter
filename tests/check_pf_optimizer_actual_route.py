"""Actual published PF Q3 route, synthetic inputs only; no H4 evaluation."""
from copy import deepcopy
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_rpe_optimizer_q3_job as J
VIEW=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-route-view-b-cbc471c91')
ROUTE=J.route(VIEW,'pf_b_pf6')
F=sys.modules['_rpe_opt_pf6']
sys.path.insert(0,str(VIEW/'tests'))
from check_pf_shortlist_q3_caller import artificial,MOCK

class ActualRoute(unittest.TestCase):
    def test_compatible_constructor_and_untouched_ast_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            F.R.Observed(Path(tmp),'ARTIFICIAL_CONSTRUCTOR',workers=1,outcome_batch=8)
        node=ast.parse(textwrap.dedent(inspect.getsource(F.R.P.R.total_state_dp))).body[0]
        actual=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
        self.assertEqual(actual,F.R.P.REFERENCE_DP_AST_SHA256)
    def test_actual_worker_persistence_and_optimizer_validator(self):
        case,inputs,binding,view=artificial();inputs['concurrent_configurations']=1
        original=deepcopy((case,inputs,view))
        with tempfile.TemporaryDirectory() as tmp,patch.object(F,'metadata',return_value=(binding,view)):
            out=Path(tmp);(out/case['label']).mkdir()
            with ROUTE.installed(out):supervision=ROUTE.supervise([case],inputs,MOCK,out,poll_seconds=.05)
            self.assertFalse(supervision['errors']);self.assertFalse(supervision['monitor_errors'])
            self.assertEqual(len(supervision['receipts']),1)
            value=ROUTE.validate(case,inputs,out/case['label'],MOCK)
            self.assertEqual(value['scientific_evaluation_calls'],1) # ARTIFICIAL toy, not target H4.
            self.assertEqual(value['status'],J.PASS)
            self.assertTrue((out/case['label']/'API_RETURN.json').exists())
            self.assertTrue((out/case['label']/'PARTIAL_TOTAL_DP.json').exists())
        self.assertEqual((case,inputs,view),original)

if __name__=='__main__':unittest.main(verbosity=2)
