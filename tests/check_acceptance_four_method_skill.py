"""Artificial failures of the four-method saved-result acceptance judge."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import acceptance_four_method_skill as T


def fixture():
    expected={'expected':{}};methods={};needs=[]
    for i,m in enumerate(T.METHODS):
        b=dict(configuration_id=m,N_X=[1]*13,N_Y=[2]*13,status='PASS',cost=i+1,MSE_lower=1,MSE_upper=2)
        row=dict(b,q3_status='PASS',conditional_cost=i+1);row.pop('cost');row.pop('status')
        expected['expected'][m]=dict(minimum=i+1,minima=[row],states={'FEASIBLE':1,'PRICE_WAITING':1})
        methods[m]=dict(best=b,stop_reason='BUDGET0',new_calls=0)
        needs.extend([dict(method=m,configuration_id=m,ready=True),dict(method=m,configuration_id=m+'_WAIT',ready=False)])
    inv=dict(budget={m:0 for m in T.METHODS},launched={m:0 for m in T.METHODS},new_calls={m:0 for m in T.METHODS})
    first=dict(result={'methods':methods},needs={'candidates':needs},invocations=[inv],jobs=[],launch_records=[],outcomes=[])
    second=deepcopy(first);second['invocations'].append(deepcopy(inv));return expected,first,second


class Checks(unittest.TestCase):
    def test_accepts_all_four_saved_methods_and_resume(self):
        self.assertTrue(all(x['status']=='PASS'for x in T.validate(*fixture())))
    def test_rejects_wrong_configuration_allocation_verdict_or_cost(self):
        for field,value in [('configuration_id','OTHER'),('N_X',[3]*13),('status','FAIL'),('cost',99)]:
            e,a,b=fixture();a['result']['methods']['PR_PAULI']['best'][field]=value
            cs=T.validate(e,a,b);self.assertTrue(any(x['status']=='FAIL'for x in cs if x['check']=='PR_PAULI_fixed_catalog_minimum'))
    def test_rejects_price_waiting_point_as_minimum(self):
        e,a,b=fixture();a['needs']['candidates'][0]['ready']=False
        self.assertTrue(any(x['status']=='FAIL'for x in T.validate(e,a,b)if x['check']=='FULL_PF_waiting_candidates_not_best'))
    def test_rejects_launch_or_call_outside_zero_budget(self):
        e,a,b=fixture();b['invocations'][-1]['launched']['PR_DF']=1;b['launch_records']=['ARTIFICIAL']
        self.assertTrue(any(x['status']=='FAIL'for x in T.validate(e,a,b)))
    def test_rejects_changed_stop_or_best_after_resume(self):
        e,a,b=fixture();b['result']['methods']['TRUNCATED_PF']['stop_reason']='CHANGED'
        self.assertTrue(any(x['status']=='FAIL'for x in T.validate(e,a,b)if x['check']=='TRUNCATED_PF_same_result_and_stop_after_resume'))
    def test_mse_bounds_must_match_or_contain_exact_saved_bounds(self):
        e,a,b=fixture();a['result']['methods']['FULL_PF']['best']['MSE_lower']=2
        self.assertTrue(any(x['status']=='FAIL'for x in T.validate(e,a,b)if x['check']=='FULL_PF_fixed_catalog_minimum'))

if __name__=='__main__':unittest.main(verbosity=2)
