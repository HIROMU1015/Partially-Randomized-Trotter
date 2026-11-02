"""Artificial selector/dependency/accounting gates for fresh H6 recovery."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import h6_pf_recovery_execution as r
import pf_common_run_contract as c
from test_h6_pf_recovery_preparation import inactive_fixture


class RecoveryExecutionTests(unittest.TestCase):
    def test_exact_missing22_selected_and_completed_price_not_recomputed(self):
        rows = inactive_fixture()['fresh_budget']['candidates']
        retained = dict(rows[0], q=4)
        plan = dict(rows=rows+[retained], recovery_ids=sorted(c.cid(x) for x in rows), compile_pool=[c.cid(x) for x in rows])
        eligible = {c.cid(x) for x in plan['rows']}
        self.assertEqual(r.remaining(plan, {c.cid(retained):dict(cosine=1, sine=1)}, eligible), rows)
        with self.assertRaises(ValueError): r.remaining(plan, {}, eligible)
        with self.assertRaises(ValueError): r.remaining(plan, {c.cid(rows[0]):{}}, eligible)

    def test_inherited_success_and_old_stop_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            for name in ('SIGNAL', 'QUALIFY'):
                c.save(p/name/'completion.json', dict(status='PASS', SOURCE='old'))
            c.save(p/'COST/STOP.json', dict(status='MANDATORY_STOP'))
            plan = dict(inherited_private=str(p), inherited_SOURCE='old',
                approval=dict(additional_internal_budget_approved=True), recovery_caps=dict(native_piece=1520, wrapper_compile=44))
            r.validate_dependencies(plan)
            plan['approval']['additional_internal_budget_approved'] = False
            with self.assertRaises(ValueError): r.validate_dependencies(plan)
            plan['approval']['additional_internal_budget_approved'] = True
            c.save(p/'COST/completion.json', dict(status='PASS'))
            with self.assertRaises(ValueError): r.validate_dependencies(plan)

    def endpoints(self):
        endpoints=[]; prices={}; pr=[]
        for f in c.FORMULAS:
            row=dict(system='H6', method='FULL', prefix=19, q=1, formula=f)
            prices[c.cid(row)]=dict(cosine=10, sine=20)
            for epsilon in c.EPS:
                endpoints.append(dict(candidate=row, epsilon=epsilon, status='ELIGIBLE', N_cos=3, N_sin=5))
                pr.append(dict(system='H6', epsilon=str(epsilon), formula=f, status='ELIGIBLE',
                    candidate_id='PR_'+f, total_RZ_numerator='143', total_RZ_denominator='1',
                    n='33', total_RZ_variance='17', total_RZ_SE='0.71'))
        return endpoints,prices,pr

    def test_fixed_accounting_preserves_inputs_and_saved_PR_statistics(self):
        ep, prices, pr=self.endpoints(); original=copy.deepcopy((ep,prices,pr)); calls=[]
        result,comps=r.account_saved(ep,prices,pr,lambda *x:calls.append(x))
        self.assertEqual((ep,prices,pr),original)
        self.assertEqual(len(comps),20)
        self.assertTrue(all(e['total_RZ_exact']=='130' for e in result))
        self.assertTrue(all(Fraction(x['PR_over_PF_exact'])==Fraction(143,130) for x in comps))
        self.assertTrue(all(abs(x['PR_reduction_percent']+10)<1e-12 for x in comps))
        self.assertTrue(all(x['PR_saved'] in pr for x in comps))
        self.assertEqual(sum(x[0]=='comparison' for x in calls),20)

    def test_missing_eligible_price_stops_before_ranking(self):
        ep,prices,pr=self.endpoints();prices.pop(c.cid(ep[0]['candidate']))
        with self.assertRaisesRegex(ValueError,'NO_MISSING_ELIGIBLE_PRICE'):
            r.account_saved(ep,prices,pr,lambda *args:None)

    def test_ineligible_endpoint_needs_no_compile_price(self):
        ep,prices,pr=self.endpoints();row=dict(ep[0]['candidate'],prefix=5,method='B0')
        for epsilon in c.EPS:ep.append(dict(candidate=row,epsilon=epsilon,status='INELIGIBLE',N_cos=None,N_sin=None))
        result,comps=r.account_saved(ep,prices,pr,lambda *args:None)
        self.assertEqual(sum(x['total_RZ_exact'] is None for x in result),5)
        self.assertTrue(all(x['PF']['status']=='COMPLETE_REGISTERED_GRID' for x in comps))


if __name__ == '__main__': unittest.main()
