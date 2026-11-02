"""Artificial joins, replay, fixed PR, rankings and fail-closed coverage."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'))
sys.path.insert(0,str(ROOT/'scripts'))
import h6_noncancellation_stage_b as c


def fixture():
    common=dict(system='H6_DF19',bias_mode=c.BIAS,epsilon='0.05',allowance_factor='1.0')
    rows=[];units={};bcid='B0I_p15_q2_2nd'
    for cid,p,q,eligible,cost,price,new in [('A','10','1',True,'100','50','70'),(bcid,'15','2',None,None,'55','55')]:
        tids={a:':'.join(('H6_DF19',cid,a)) for a in c.old.AXES}
        rows.append(dict(common,row_id=cid,candidate_id=cid,method='B0' if cid=='A' else 'B0I',kind='PF',
            prefix=p,q=q,PF='2nd',eligible=eligible,saved_total_RZ_literal=cost,
            saved_shots={'cosine':1,'sine':1} if eligible else {},
            saved_bias_and_shot_policy={'artificial':True} if eligible else None,
            saved_cost_status='SAVED',unit_ids=tids if eligible else {},
            source_id='ARTIFICIAL',source_locator=cid,source_file_sha256='0'*64))
        for a,tid in tids.items():units[tid]=dict(task_id=tid,baseline_RZ_literal=price,replacement_RZ_literal=new,
            ALL_SAVED458='REPLACE_NEW450' if cid=='A' else 'RETAIN_OLD_MEASURED')
    rows.append(dict(common,row_id='PR',candidate_id='PR',kind='PR',eligible=True,
        saved_total_RZ_literal='105',saved_PR_SE_literal='2.5',paired_samples=32,cohort='CONFIRMATION_UNION8_N32'))
    a=[dict(candidate_id=bcid,prefix=15,q=2,PF='2nd',T=.8,bias_mode=c.BIAS,epsilon=.05,allowance_factor=1.,
            eligible=True,shots={'cosine':1,'sine':1},accounting=dict(cost_acquisition_eligible=True,
                axes={'real':{'shots':1},'imag':{'shots':1}}))]
    condition=dict(common,original_scope='toy',candidate_ids=['A',bcid],old_profile_pending_axes_in_scope=0)
    saved=dict(common,scope='toy',saved_result=dict(deterministic=dict(status='UNRESOLVED_FULL_DOMAIN',best=None),
            confirmed_PR=dict(best=dict(cell_id='PR',G_exact='105'))))
    return rows,a,units,[condition],[saved]


def execute_toy():
    original,a,units,conditions,saved=fixture()
    rows,lineage=c.join_stage_a(original,a,{'SOURCE':'ARTIFICIAL'},artificial=True)
    costs,gate=c.baseline(rows,units);gates=c.baseline_selections(costs,rows,saved,conditions)
    c.old.replace_costs(costs,units);results,ranks,changes=c.compare(costs,conditions,gates)
    audit=c.audit(costs,rows,units,conditions,results,ranks,changes)
    return original,rows,lineage,costs,gate,results,ranks,changes,audit


class StageBTests(unittest.TestCase):
    def test_new_scoped_grant_required(self):
        with self.assertRaises(ValueError):c.authorize({}, {})
        c.authorize({'kind':'ARTIFICIAL'},{},artificial=True)
        with self.assertRaises(ValueError):c.authorize({'kind':'REAL'},{},artificial=True)

    def test_only_unknown_row_is_recovered_originals_immutable(self):
        rows,a,_,_,_=fixture();before=deepcopy(rows);out,lineage=c.join_stage_a(rows,a,{},artificial=True)
        self.assertEqual(rows,before);self.assertEqual(out[0],before[0]);self.assertEqual(out[2],before[2])
        self.assertEqual(len(lineage),1);self.assertIsNone(out[1]['saved_total_RZ_literal'])

    def test_stageA_unknown_cannot_be_excluded_as_false(self):
        rows,a,_,_,_=fixture();a[0]['eligible']=None
        with self.assertRaisesRegex(ValueError,'UNRESOLVED'):c.join_stage_a(rows,a,{},artificial=True)

    def test_extra_or_missing_patch_refused(self):
        rows,a,_,_,_=fixture()
        with self.assertRaisesRegex(ValueError,'EXACT_UNKNOWN'):c.join_stage_a(rows,[],{},artificial=True)
        with self.assertRaisesRegex(ValueError,'DUPLICATE_STAGE'):c.join_stage_a(rows,a+a,{},artificial=True)

    def test_known_row_cannot_be_reclassified(self):
        rows,a,_,_,_=fixture();rows[1]['eligible']=False
        with self.assertRaisesRegex(ValueError,'PATCH_ONLY'):c.join_stage_a(rows,a,{},artificial=True)

    def test_saved_stageA_shots_must_match_accounting(self):
        rows,a,_,_,_=fixture();a[0]['shots']['cosine']=2
        with self.assertRaisesRegex(ValueError,'SAVED_SHOTS'):c.join_stage_a(rows,a,{},artificial=True)

    def test_known_baseline_mismatch_stops(self):
        rows,a,units,_,_=fixture();rows[0]['saved_total_RZ_literal']='101'
        joined,_=c.join_stage_a(rows,a,{},artificial=True)
        with self.assertRaisesRegex(ValueError,'BASELINE_COST_MISMATCH'):c.baseline(joined,units)

    def test_new_cost_has_distinct_provenance_and_no_fake_replay(self):
        *_,costs,gate,result,ranks,changes,audit=execute_toy()
        self.assertEqual(gate['known_costs_replayed'],1);self.assertEqual(gate['newly_constructed_eligible_baseline_costs'],1)
        self.assertIsNone(costs[1]['baseline_replay_matches']);self.assertFalse(result[0]['baseline_saved_result_matches'])

    def test_new_winner_q_and_PR_direction_can_change(self):
        result=execute_toy()[5][0]
        self.assertEqual(result['baseline']['candidate_id'],'A')
        self.assertEqual(result['replacement']['candidate_id'],'B0I_p15_q2_2nd')
        self.assertTrue(result['q_changed'] and result['PR_PF_relation_changed'])
        self.assertEqual(result['saved_PR_total_exact'],'105');self.assertEqual(result['saved_PR_SE_literal'],'2.5')

    def test_all_integer_checks_and_pair_inversion(self):
        *_,changes,audit=execute_toy()
        self.assertEqual(audit['weighted_cost_checks'],4);self.assertEqual(audit['pair_relations_checked'],1)
        self.assertEqual(len(changes),1);self.assertTrue(changes[0]['strict_inversion'])

    def test_missing_price_never_zero_filled(self):
        rows,a,units,_,_=fixture();units['H6_DF19:B0I_p15_q2_2nd:cosine']['baseline_RZ_literal']=None
        joined,_=c.join_stage_a(rows,a,{},artificial=True)
        with self.assertRaises(ValueError):c.baseline(joined,units)

    def test_registered_candidate_missing_blocks_minimum(self):
        rows,a,units,conditions,saved=fixture();joined,_=c.join_stage_a(rows,a,{},artificial=True);costs,_=c.baseline(joined,units)
        conditions[0]['candidate_ids'].append('MISSING')
        with self.assertRaisesRegex(ValueError,'REGISTERED_DOMAIN'):c.baseline_selections(costs,joined,saved,conditions)

    def test_saved_PR_cost_conflict_stops(self):
        rows,a,units,conditions,saved=fixture();rows[-1]['saved_total_RZ_literal']='106'
        joined,_=c.join_stage_a(rows,a,{},artificial=True);costs,_=c.baseline(joined,units)
        with self.assertRaisesRegex(ValueError,'PR_BINDING'):c.baseline_selections(costs,joined,saved,conditions)

    def test_old70_rows_copied_no_numeric_cost_replay(self):
        results=execute_toy()[5];old=deepcopy(results);old[0]['condition_id']='OTHER';old[0]['saved_PR_total_exact']='not re-evaluated'
        before=deepcopy(old);allrows,summary=c.aggregate(results,old)
        self.assertEqual(old,before);self.assertEqual(allrows[0]['saved_PR_total_exact'],'not re-evaluated')
        self.assertFalse(summary['old70_costs_choices_ratios_or_ranks_recalculated'])


if __name__=='__main__':unittest.main()
