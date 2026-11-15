"""Artificial metadata checks; no physical price or expectation calculation."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

import admit_full_h_d7_holdout as C


def comparison_fixture():
    loc=dict(commit='artificial',path='receipt',selector=[0])
    p=dict(cell_id='M6_ROUND',axis='X',physical={'id':'mock'},cohort='mock',step_seeds=[[17]],
           n_paths=1,predicted_RZ={'numerator':123,'denominator':2})
    u={k:deepcopy(p[k]) for k in ('cell_id','axis','physical','cohort','step_seeds','n_paths')}
    u.update(P=deepcopy(p['predicted_RZ']),O={'numerator':62,'denominator':1},
             signed_error={'supplied':'signed'},signed_relative_error={'supplied':'relative'},
             status='VALID',model_verdict=C.VERDICT,tolerance=None,observation_sources=[loc])
    receipts=[dict(status='ACCEPTED_SAVED_RECEIPT',original_record=dict(index=0,locators=loc))]
    return [u,deepcopy(u),p,receipts]


def schedule_fixture():
    case={'rows':[]};joined=dict(model_status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD',rows=[],
        original_configuration_id='original',total_configuration_id='total',binding_id='bridge',
        allocation=dict(allocation_id='fixed',shots=[]));alignment={'rows':[]}
    for m in range(13):
        p=dict(id='physical_'+str(m),family='FULL_H_PURE_RTE',L_D=0,K=2,q=1,R_outer=3,s=3)
        case['rows'].append(dict(m=m,physical_candidate=p,T={'supplied':m},raw_record_source={'supplied':m}))
        joined['allocation']['shots'].append(dict(m=m,N_X=C.SHOTS[m],N_Y=C.SHOTS[m]))
        for axis in ('X','Y'):
            b=dict(m=m,axis=axis,physical_id=p['id'],N=C.SHOTS[m],original_candidate_id=f'o{m}{axis}',
                   total_candidate_id=f't{m}{axis}',original_law_selector=['original',m],total_law_selector=['total',m])
            joined['rows'].append(b)
            alignment['rows'].append(dict(physical_id=p['id'],axis=axis,physical_candidate=p,
                configuration_id='original',original_canonical_id=b['original_candidate_id'],
                decoder=dict(total_canonical_id=b['total_candidate_id']),price=[]))
    frozen=dict(models={a:dict(input_locators=[{'source':'artificial calibration'}]) for a in ('X','Y')})
    return [case,joined,alignment,frozen,[],[]]


class Tests(unittest.TestCase):
    def test_saved_comparison_is_copied_without_arithmetic(self):
        d=comparison_fixture();out=C.check_comparison(*d)
        self.assertEqual(out,{k:d[0][k] for k in C.EXACT_KEYS})
        self.assertIsNot(out['P'],d[0]['P'])

    def test_wrong_prediction_source_axis_or_tolerance_rejected(self):
        for key,value in [('predicted_RZ',{'numerator':999,'denominator':1}),('axis','Y')]:
            d=comparison_fixture();d[2][key]=value
            with self.assertRaises(ValueError):C.check_comparison(*d)
        d=comparison_fixture();d[0]['tolerance']=.02;d[1]=deepcopy(d[0])
        with self.assertRaisesRegex(ValueError,'tolerance'):C.check_comparison(*d)

    def test_held_or_duplicate_paths_not_accepted_as_comparison(self):
        d=comparison_fixture();d[3][0]['status']='HELD'
        with self.assertRaisesRegex(ValueError,'receipts'):C.check_comparison(*d)
        d=comparison_fixture();d[3].append(deepcopy(d[3][0]))
        with self.assertRaisesRegex(ValueError,'path count'):C.check_comparison(*d)

    def test_26_axis_allocation_join_keeps_unknown_expectations(self):
        rows=C.join_rows(*schedule_fixture())
        self.assertEqual(len(rows),26)
        self.assertEqual([r['N'] for r in rows if r['axis']=='X'],C.SHOTS)
        self.assertTrue(all(r['E_C'] is None and r['CI'] is None and r['G'] is None for r in rows))
        self.assertTrue(all(r['qualified_for_candidate'] is False for r in rows))
        self.assertTrue(all(r['q_PF'] is None for r in rows))

    def test_physical_or_allocation_mismatch_stops_join(self):
        d=schedule_fixture();d[1]['rows'][0]['N']=1
        with self.assertRaisesRegex(ValueError,'allocation'):C.join_rows(*d)
        d=schedule_fixture();d[1]['rows'][0]['physical_id']='other'
        with self.assertRaisesRegex(ValueError,'physical'):C.join_rows(*d)
        d=schedule_fixture();d[1]['rows'][-1]=deepcopy(d[1]['rows'][0])
        with self.assertRaisesRegex(ValueError,'Unique'):C.join_rows(*d)

    def test_entrypoint_contains_no_numerical_science_calls(self):
        tree=ast.parse(Path(C.__file__).read_text())
        calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
               for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(calls & {'fit','predict','mean','evaluate','sample_requests','transpile','compile','cost','calculate'})
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_comparison')
        self.assertFalse(any(isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Sub,ast.Div,ast.Mult,ast.Pow)) for n in ast.walk(fn)))


if __name__=='__main__':
    unittest.main()
