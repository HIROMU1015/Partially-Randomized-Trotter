"""Artificial saved rows only; never load a real P result or reaccount costs."""
from copy import deepcopy
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_h2_p_completed as audit
import h2_p_contract as c


class SavedPointRows(unittest.TestCase):
    def setUp(self):
        self.endpoints=[dict(cell_id='toy'+str(i),epsilon=e,n=n,PF=dict(G_RZ_exact='100'),
            saved_shots=dict(cosine=3,sine=5),expected_direction=1) for i,(e,n) in enumerate(((.05,80),(.0005,12)))]
        self.rows=[dict(cell_id=e['cell_id'],epsilon=e['epsilon'],n=e['n'],PF=deepcopy(e['PF']),
            shots=deepcopy(e['saved_shots']),G_exact='20',sample_variance_exact='4',sample_SD=2.,paired_SE=1.,
            point_direction=1,status='REPRODUCED',population_mean_certified=False) for e in self.endpoints]
        self.account=dict(comparisons=self.rows,candidate_reselected=False,old_cohorts_pooled=False,
            population_mean_certified=False,actual_power_certified=False,law_probabilities=None,
            unseen_event_cost=None,unseen_event_variance=None)

    def test_saved_comparisons_accepted_with_math_functions_disabled(self):
        with (patch.object(c.base.stats,'evaluate_pairs',side_effect=AssertionError('REACCOUNT')),
              patch.object(c.base.stats,'normal_n_formula',side_effect=AssertionError('NEW_N'))):
            audit.check_saved_accounting(self.account,self.endpoints)

    def test_missing_extra_or_reordered_endpoints_refused(self):
        for rows in (self.rows[:1],self.rows+self.rows[:1],list(reversed(self.rows))):
            with self.assertRaises(ValueError):audit.check_saved_accounting(dict(self.account,comparisons=rows),self.endpoints)

    def test_n_PF_or_shot_substitution_refused(self):
        for key,value in [('n',81),('PF',dict(G_RZ_exact='101')),('shots',dict(cosine=4,sine=5))]:
            r=deepcopy(self.account);r['comparisons'][0][key]=value
            with self.assertRaises(ValueError):audit.check_saved_accounting(r,self.endpoints)

    def test_adverse_direction_and_tie_are_valid_saved_outcomes(self):
        for direction,status in [(-1,'NOT_REPRODUCED'),(0,'INCONCLUSIVE')]:
            self.rows[0].update(point_direction=direction,status=status)
            audit.check_saved_accounting(self.account,self.endpoints)

    def test_inconsistent_status_refused(self):
        self.rows[0]['point_direction']=-1
        with self.assertRaises(ValueError):audit.check_saved_accounting(self.account,self.endpoints)

    def test_population_or_unknown_tail_filled_in_refused(self):
        for change in [dict(population_mean_certified=True),dict(actual_power_certified=True),dict(unseen_event_cost=0),dict(unseen_event_variance=0),dict(law_probabilities={})]:
            with self.assertRaises(ValueError):audit.check_saved_accounting(dict(self.account,**change),self.endpoints)

    def test_nonfinite_saved_statistics_refused(self):
        self.rows[0]['paired_SE']=float('nan')
        with self.assertRaises(ValueError):audit.check_saved_accounting(self.account,self.endpoints)

    def test_public_projection_removes_master_and_axis_seeds(self):
        pair=dict(cell_id='toy',seed=123,replica=0,axes={a:dict(seed=123,RZ=2,event_digest='toy') for a in ('cosine','sine')})
        before=deepcopy(pair);p=audit.public_pair(pair)
        self.assertNotIn('seed',p);self.assertTrue(all('seed' not in v for v in p['axes'].values()))
        self.assertEqual(before,pair)

    def test_auditor_has_no_statistical_or_scientific_dispatch(self):
        tree=ast.parse(Path(audit.__file__).read_text())
        names={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree)
            if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
        self.assertTrue(names.isdisjoint({'account','evaluate_pairs','normal_n_formula','compile_one','steps','load','analyze'}))


if __name__=='__main__':unittest.main()
