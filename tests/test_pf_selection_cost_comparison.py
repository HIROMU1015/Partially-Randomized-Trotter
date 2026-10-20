"""Artificial contract, replay-gate, exact ranking and fixed-PR tests."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'))
sys.path.insert(0,str(ROOT/'scripts'))
import pf_selection_cost_comparison as c
from run_pf_selection_cost_comparison import selected_conditions


class ComparisonTests(unittest.TestCase):
    def fixture(self):
        common=dict(system='H6_DF19',bias_mode='signed_signal',epsilon='0.1',allowance_factor='1.0')
        rows=[];units={}
        for cid,price,new,q in [('A','50','60','1'),('B','55','55','2')]:
            ids={a:cid+':'+a for a in c.AXES}
            rows.append(dict(common,row_id=cid,candidate_id=cid,kind='PF',method='B0',prefix='1',q=q,PF='2nd',
                eligible=True,saved_shots={'cosine':1,'sine':1},saved_total_RZ_literal=str(2*Fraction(price)),
                source_id='ARTIFICIAL',source_locator=cid,source_file_sha256='0'*64,unit_ids=ids))
            for axis in c.AXES:
                units[ids[axis]]=dict(task_id=ids[axis],baseline_RZ_literal=price,replacement_RZ_literal=new,
                    ALL_SAVED458='REPLACE_NEW450' if cid=='A' else 'RETAIN_OLD_MEASURED')
        pr=dict(common,row_id='PR',candidate_id='PR',kind='PR',eligible=True,paired_samples=32,
            saved_total_RZ_literal='105',saved_PR_SE_literal='1.5',cohort='ARTIFICIAL_N32')
        rows.append(pr)
        condition=dict(common,original_scope='old32',candidate_ids=['A','B'],old_profile_pending_axes_in_scope=0)
        saved=dict(common,scope='old32',saved_result={
            'deterministic':dict(best=dict(cell_id='A',G_exact='100'),exact_ties=['A']),
            'confirmed_PR':dict(best=dict(cell_id='PR',G_exact='105'),exact_ties=['PR']),
            'PR_over_deterministic_exact':'21/20'})
        return rows,units,[condition],[saved]

    def execute_toy(self):
        rows,units,conditions,saved=self.fixture()
        costs=c.replay_pf(rows,units,{c.endpoint(conditions[0])})
        gate=c.baseline_gate(costs,rows,saved,conditions)
        c.replace_costs(costs,units)
        comparisons,ranks,changes=c.compare(costs,conditions,gate)
        return rows,costs,comparisons,ranks,changes

    def test_new_grant_required_and_synthetic_permit_typed(self):
        with self.assertRaises(ValueError):c.authorize({}, {})
        with self.assertRaises(ValueError):c.authorize({'kind':'SAVED_REAL'}, {},synthetic=True)
        c.authorize({'kind':'ARTIFICIAL'}, {},synthetic=True)

    def test_axis_weighting_is_exact(self):
        self.assertEqual(c.weighted({'cosine':2,'sine':3},{'cosine':'1/3','sine':'2/7'}),Fraction(32,21))

    def test_missing_zero_and_bad_shots_refused(self):
        for price in (None,'0','-1'):
            with self.assertRaises((ValueError,TypeError)):c.weighted({'cosine':1,'sine':1},{'cosine':price,'sine':'1'})
        for value in (0,-1,True,1.5):
            with self.assertRaises(ValueError):c.weighted({'cosine':value,'sine':1},{'cosine':'1','sine':'1'})

    def test_baseline_mismatch_stops_before_replacement(self):
        rows,units,conditions,_=self.fixture();rows[0]['saved_total_RZ_literal']='101'
        with self.assertRaisesRegex(ValueError,'BASELINE_COST_MISMATCH'):
            c.replay_pf(rows,units,{c.endpoint(conditions[0])})

    def test_saved_winner_and_ratio_mismatch_rejected(self):
        rows,units,conditions,saved=self.fixture();costs=c.replay_pf(rows,units,{c.endpoint(conditions[0])})
        saved[0]['saved_result']['PR_over_deterministic_exact']='1'
        with self.assertRaisesRegex(ValueError,'SAVED_BASELINE_RATIO_MISMATCH'):c.baseline_gate(costs,rows,saved,conditions)

    def test_new_pf_q_and_PR_relation_can_change_with_fixed_PR(self):
        rows,costs,comp,_,changes=self.execute_toy();r=comp[0]
        self.assertEqual(r['baseline']['candidate_id'],'A');self.assertEqual(r['replacement']['candidate_id'],'B')
        self.assertTrue(r['q_changed'] and r['best_candidate_changed'] and r['PR_PF_relation_changed'])
        self.assertEqual(r['saved_PR_total_exact'],'105');self.assertEqual(r['saved_PR_SE_literal'],'1.5')
        self.assertEqual(r['PR_PF_relation_baseline'],'PF_LOWER');self.assertEqual(r['PR_PF_relation_replacement'],'PR_LOWER')
        self.assertEqual(len(changes),1);self.assertTrue(changes[0]['strict_inversion'])

    def test_old_measured_unit_cannot_change(self):
        rows,units,conditions,_=self.fixture();costs=c.replay_pf(rows,units,{c.endpoint(conditions[0])})
        units['B:cosine']['replacement_RZ_literal']='56'
        with self.assertRaisesRegex(ValueError,'OLD_MEASURED_UNIT_CHANGED'):c.replace_costs(costs,units)

    def test_excluded_endpoint_never_evaluated(self):
        rows,units,conditions,_=self.fixture();bad=deepcopy(rows[0]);bad.update(row_id='bad',bias_mode='empirical_no_cancellation',saved_total_RZ_literal='invalid')
        result=c.replay_pf(rows+[bad],units,{c.endpoint(conditions[0])})
        self.assertEqual(len(result),2)

    def test_input_rows_not_mutated(self):
        rows,units,conditions,saved=self.fixture();before=deepcopy((rows,units,saved))
        costs=c.replay_pf(rows,units,{c.endpoint(conditions[0])});gate=c.baseline_gate(costs,rows,saved,conditions)
        c.replace_costs(costs,units);c.compare(costs,conditions,gate)
        self.assertEqual((rows,units,saved),before)

    def test_tie_change_distinguished_from_inversion(self):
        rows,units,conditions,saved=self.fixture()
        for axis in c.AXES:units['A:'+axis]['replacement_RZ_literal']='55'
        costs=c.replay_pf(rows,units,{c.endpoint(conditions[0])});gate=c.baseline_gate(costs,rows,saved,conditions)
        c.replace_costs(costs,units);results,ranks,changes=c.compare(costs,conditions,gate)
        self.assertEqual(results[0]['replacement']['exact_ties'],['A','B'])
        self.assertEqual(results[0]['replacement']['candidate_id'],'A')
        self.assertEqual(results[0]['strict_pairwise_inversions'],0);self.assertEqual(results[0]['pairwise_tie_changes'],1)
        self.assertEqual([r['replacement']['cost_tie_rank'] for r in ranks],[1,1])

    def test_positive_ratio_three_states_exact(self):
        self.assertEqual(c.relation(Fraction(999,1000)),'PR_LOWER')
        self.assertEqual(c.relation(Fraction(1)),'TIE')
        self.assertEqual(c.relation(Fraction(1001,1000)),'PF_LOWER')

    def test_unresolved_is_not_ineligible(self):
        rows,units,conditions,_=self.fixture();rows[0]['eligible']=None
        with self.assertRaisesRegex(ValueError,'UNRESOLVED'):c.replay_pf(rows,units,{c.endpoint(conditions[0])})

    def test_real_preparation_scope_structure_only(self):
        import json
        p=ROOT/'artifacts/resource_applicability/track_a_pf_selection_comparison_preparation/2026-10-20'
        data={n:json.loads((p/n).read_text()) for n in ('condition_coverage.json','candidate_domains.json')}
        selected,excluded=selected_conditions(data)
        self.assertEqual(len(selected),70);self.assertEqual(len(excluded),30)
        self.assertEqual(len({c.endpoint(r) for r in selected}),30)


if __name__=='__main__':unittest.main()
