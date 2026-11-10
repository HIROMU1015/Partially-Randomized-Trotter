"""Artificial finite-R menus only. Never use molecular N/C values to rank R."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests'),str(ROOT/'scripts')]
from trottertracks import rpe_r_selection as r
from check_pf_pr_rpe_pipeline import fixture, fake_shot

p=r.p;a=r.a


def inputs():
    task,_,_,_=fixture();task['binding']['system']='H4'
    for row in task['candidates']:
        c=row['candidate'];c.update(system='H4',prefix=7 if p.arm(c)=='PF' else 1,
            q=1<<row['m'],delta=.1,method='B0' if p.arm(c)=='PF' else 'PR')
        c['id']=p.candidate_id(c);row['q_PF']=c['q'];row['structure']=p.structure(c)
    menu=r.build_menu(task,{0:[4,8],1:[4,8]},sampling_law=dict(model='ideal_finite_RTE_law',
        sampling='MOCK_IDEAL_IID',source={'fixture':'MOCK_LAW_DEFINITION'}))
    ns=[];cs=[];select={}
    for row in menu['candidates']:
        n=fake_shot(menu,row);n['law_binding']=r.request_context(menu,row)['law_binding']
        n['diagnostics']={'fixture':'MOCK_PRECISION_RADIUS_PHASE_ONLY'}
        arm=p.arm(row['candidate']);R=row['candidate']['R_outer']
        pair=(3,4) if arm=='PF' else ((20,20) if R==4 else (2,3)) if row['m']==0 else ((2,2) if R==4 else (1,1))
        n['plan'].update(N_X=pair[0],N_Y=pair[1]);ns.append(n)
        for axis in p.AXES:
            v=dict(**p.measurement(row,axis),**r.request_context(menu,row),axis=axis,cohort='MOCK_SCENARIO',n_paths=999,
                mean={'rz_count':(10 if axis=='X' else 11) if arm=='PF' or R==8 else (1 if axis=='X' else 2)},
                source={'fixture':'MOCK_C_ONLY'},normalization_is_price_multiplier=False,peak_qubits=None,
                classification='DIRECT_COMPILED_DETERMINISTIC' if arm=='PF' else 'MODEL_PREDICTION',
                expectation='DETERMINISTIC' if arm=='PF' else 'QUALIFIED_EXPECTATION',
                model=None if arm=='PF' else dict(qualified_for_candidate=True,q_variable='q_PF',q=row['q_PF'],
                    source={'fixture':'MOCK_MODEL'},validation_source={'fixture':'MOCK_VALIDATION'},coverage='MOCK_R_DOMAIN',
                    domain=deepcopy(row['candidate']),binding=deepcopy(menu['binding']),compiler=menu['profile']['compiler']),
                uncertainty=dict(shared_calibration_group='MOCK_SHARED',coefficient_uncertainty=None,
                    long_q_systematic=None,joint_covariance=None,total_CI=None))
            cs.append(v);select[row['candidate']['id'],axis]=v['cohort']
    return task,menu,ns,cs,select


class RSelectionTests(unittest.TestCase):
    def test_R_only_menu_preserves_time_q_delta_and_policy(self):
        task,menu,ns,cs,select=inputs()
        self.assertEqual(menu['configuration'],task['configuration'])
        for row in menu['candidates']:
            old=next(v for v in task['candidates'] if v['m']==row['m'] and p.arm(v['candidate'])==p.arm(row['candidate']))
            for key in ['T','q','delta','prefix','K','formula']:
                self.assertEqual(row['candidate'][key],old['candidate'][key])
            self.assertEqual(row['q_RPE'],old['q_RPE'])
            self.assertEqual(row['structure']['event_count'],row['q_PF']*(row['candidate']['R_outer'] or 0))
        self.assertEqual(len(menu['candidates']),6)
        for bad in [[4,4],[True],[0],[],[4.]]:
            with self.assertRaises(ValueError):r.build_menu(task,{0:bad},sampling_law=menu['sampling_law'])

    def test_H6_signed_allocation_reused_and_early_delta_not_repurposed(self):
        task,menu,_,_,_=inputs();task=deepcopy(task)
        task['binding']['system']='H6';task['configuration']['horizon']=dict(maximum_round_index_M=4,round_count_M_plus_one=5)
        task['configuration']['alpha_axis']=.05/10
        for row in task['candidates']:
            c=row['candidate'];c.update(system='H6',formula='4th(m5_best)',prefix=19 if p.arm(c)=='PF' else 7,
                method='FULL' if p.arm(c)=='PF' else 'PR',R_outer=None if p.arm(c)=='PF' else 16)
            row.update(m=4,q_RPE=16,q_PF=1);c.update(T=1.6,q=1,delta=1.6)
            c['id']=p.candidate_id(c);row['structure']=p.structure(c)
        task['candidates']=task['candidates'][:2]
        expanded=r.build_menu(task,{4:[16,32]},sampling_law=menu['sampling_law'])
        pr=[v for v in expanded['candidates'] if p.arm(v['candidate'])=='PR']
        self.assertEqual([sum(v['structure']['allocation']) for v in pr],[16,32])
        self.assertTrue(all(len(v['structure']['signed_weights'])==11 for v in pr))
        with self.assertRaises(ValueError):r.build_menu(task,{4:[1]},sampling_law=menu['sampling_law'])
        for row in task['candidates']:
            row['candidate'].update(delta=.8,q=2);row['q_PF']=2
            row['candidate']['id']=p.candidate_id(row['candidate']);row['structure']=p.structure(row['candidate'])
        with self.assertRaisesRegex(ValueError,'delta'):r.build_menu(task,{4:[16,32]},sampling_law=menu['sampling_law'])

    def test_point_tradeoff_selects_different_R_by_round_and_reuses_accounting(self):
        _,menu,ns,cs,select=inputs()
        with patch.object(a,'aggregate_supplied_axes',wraps=a.aggregate_supplied_axes) as aggregate:
            out=r.evaluate_menu(menu,ns,cs,cohort_selection=select)
        self.assertEqual(aggregate.call_count,8)
        self.assertEqual([v['R_outer'] for v in out['selections'] if v['arm']=='PR'],[8,4])
        self.assertEqual(a.exact_scalar(out['totals']['PR']['rz_count']),59)
        self.assertEqual(a.exact_scalar(out['totals']['PF']['rz_count']),148)
        self.assertEqual(out['comparison']['verdict'],'POINT_ESTIMATES_ONLY')
        self.assertTrue(out['menu_complete']);self.assertFalse(out['scientific_R_adopted'])
        self.assertFalse(out['rpe_success_qualified']);self.assertFalse(out['normalization_is_price_multiplier'])
        self.assertFalse(out['classical_samples_are_shot_multipliers'])

    def test_huge_integer_one_unit_order_and_binary64_fraction_are_exact(self):
        _,menu,ns,cs,select=inputs();huge=(1<<1200)+3
        for n in ns:
            R=n['identity']['candidate']['R_outer']
            n['plan'].update(N_X=huge+(1 if R==4 else 0),N_Y=huge)
        for v in cs:v['mean']['rz_count']=1.
        out=r.evaluate_menu(menu,ns,cs,cohort_selection=select)
        self.assertEqual([v['R_outer'] for v in out['selections'] if v['arm']=='PR'],[8,8])
        self.assertEqual(a.exact_scalar(out['totals']['PR']['rz_count']),4*huge)
        self.assertEqual(json.loads(json.dumps(out))['totals']['PR']['rz_count']['numerator'],4*huge)
        for v in cs:v['mean']['rz_count']=.1
        out=r.evaluate_menu(menu,ns,cs,cohort_selection=select)
        self.assertEqual(a.exact_scalar(out['totals']['PR']['rz_count']),4*huge*Fraction.from_float(.1))

    def test_identity_and_sampling_law_mismatches_raise_before_costs(self):
        _,menu,ns,cs,select=inputs()
        for change in [lambda v:v['identity']['candidate'].update(R_outer=32),
                lambda v:v['identity']['candidate'].update(T=99),lambda v:v['identity'].update(q_PF=99),
                lambda v:v['identity']['binding'].update(state='FOREIGN'),
                lambda v:v['identity']['binding'].update(snapshot='FOREIGN'),
                lambda v:v['identity']['profile'].update(compiler='FOREIGN'),
                lambda v:v['identity']['candidate'].update(prefix=99,K=0,delta=.2),
                lambda v:v['law_binding']['definition'].update(model='FOREIGN_LAW')]:
            bad=deepcopy(cs);change(bad[1])
            with self.assertRaises(ValueError):r.evaluate_menu(menu,ns,bad,cohort_selection=select)
        bad=deepcopy(ns);bad[0]['law_binding']['q_PF']=7
        with self.assertRaisesRegex(ValueError,'law'):r.coverage(menu,bad,cs,cohort_selection=select)

    def test_missing_unknown_and_failed_candidates_are_distinct(self):
        _,menu,ns,cs,select=inputs()
        null=deepcopy(cs);null[2]['mean']['rz_count']=None
        out=r.evaluate_menu(menu,ns,null,cohort_selection=select)
        self.assertFalse(out['menu_complete']);self.assertTrue(out['schedule_complete'])
        self.assertIsNotNone(out['totals']);self.assertEqual(out['selection_scope'],'AVAILABLE_QUALIFIED_SUBSET')
        missing=r.evaluate_menu(menu,ns[1:],cs,cohort_selection=select)
        self.assertIsNone(missing['totals']);self.assertFalse(missing['schedule_complete'])
        bad=deepcopy(ns);bad[1]['plan'].update(status='UNQUALIFIED_ROUND',N_X=None,N_Y=None)
        out=r.evaluate_menu(menu,bad,cs,cohort_selection=select)
        self.assertTrue(out['menu_complete'])
        excluded=next(v for v in out['rows'] if v['candidate_id']==bad[1]['identity']['candidate']['id'])
        self.assertIn('N_PRECISION_OR_RADIUS_GATE_FAILED',excluded['reasons']);self.assertIsNone(excluded['cost'])

    def test_fixed_R_model_cannot_be_relabeled_for_another_R(self):
        _,menu,ns,cs,select=inputs()
        for v in cs:
            if v['model'] and v['identity']['candidate']['R_outer']==8:
                v['model']['domain']['R_outer']=4
        out=r.evaluate_menu(menu,ns,cs,cohort_selection=select)
        self.assertEqual([v['R_outer'] for v in out['selections'] if v['arm']=='PR'],[4,4])
        self.assertFalse(out['menu_complete'])
        self.assertTrue(any('MISSING_OR_FOREIGN_R_MODEL_DOMAIN_X' in v['reasons'] for v in out['rows']))

    def test_finite_cohort_path_and_unvalidated_extrapolation_do_not_become_law_prices(self):
        _,menu,ns,cs,select=inputs()
        for v in cs:
            if v['model']:
                v['expectation']='FINITE_COHORT_ONLY';v['law_context'].update(price_law='MOCK_FINITE_COHORT',ideal_sampler_distance_qualified=False)
        self.assertIsNone(r.evaluate_menu(menu,ns,cs,cohort_selection=select)['totals'])
        out=r.evaluate_menu(menu,ns,cs,cohort_selection=select,price_policy='finite_cohort_scenario')
        self.assertIsNotNone(out['totals']);self.assertFalse(out['full_RPE_qualified'])
        self.assertEqual(out['inventory']['price_policy'],'finite_cohort_scenario')
        for v in cs:
            if v['model']:v['expectation']='FINITE_PATH_ONLY'
        self.assertIsNone(r.evaluate_menu(menu,ns,cs,cohort_selection=select,price_policy='finite_cohort_scenario')['totals'])
        for v in cs:
            if v['model']:v['classification']='UNVALIDATED_EXTRAPOLATION'
        self.assertIsNone(r.evaluate_menu(menu,ns,cs,cohort_selection=select,price_policy='finite_cohort_scenario')['totals'])

    def test_paired_price_axes_and_legacy_law_packets_are_not_silently_accepted(self):
        _,menu,ns,cs,select=inputs();cs[3]['cohort']='OTHER';select[cs[3]['identity']['candidate']['id'],'Y']='OTHER'
        out=r.evaluate_menu(menu,ns,cs,cohort_selection=select)
        self.assertTrue(any('UNPAIRED_XY_PRICE_SCENARIO' in v['reasons'] for v in out['rows']))
        for n in ns:n.pop('law_binding')
        for v in cs:v.pop('law_binding')
        self.assertIsNone(r.evaluate_menu(menu,ns,cs,cohort_selection=select)['totals'])

    def test_bounds_are_supplied_simultaneous_and_do_not_invent_CI(self):
        _,menu,ns,cs,select=inputs()
        labels=[r._label(row,axis) for row in menu['candidates'] for axis in p.AXES]
        evidence=dict(labels=labels,source={'fixture':'MOCK_SIMULTANEOUS_BOUNDS'},covers_stat_model_law_and_menu=True,
            bounds=[[v['mean']['rz_count'],v['mean']['rz_count']] for v in cs])
        out=r.evaluate_menu(menu,ns,cs,cohort_selection=select,simultaneous_bounds=evidence)
        self.assertTrue(out['uncertainty']['strict_menu_dominance'])
        self.assertIsNone(out['uncertainty']['total_confidence_interval'])
        evidence['bounds']=[[0,100] for _ in labels]
        self.assertFalse(r.evaluate_menu(menu,ns,cs,cohort_selection=select,simultaneous_bounds=evidence)['uncertainty']['strict_menu_dominance'])
        evidence['covers_stat_model_law_and_menu']=False
        with self.assertRaises(ValueError):r.evaluate_menu(menu,ns,cs,cohort_selection=select,simultaneous_bounds=evidence)

    def test_full_RPE_success_is_separate_and_bound_to_selected_R_schedule(self):
        _,menu,ns,cs,select=inputs();out=r.evaluate_menu(menu,ns,cs,cohort_selection=select)
        evidence=dict(context=out['schedule_success_context'],source={'fixture':'MOCK_SUCCESS_ONLY'},
            prior_alias=True,horizon_decoder=True,all_round_phase_radius=True,failure_budget=True,implementation_and_law=True)
        out=r.evaluate_menu(menu,ns,cs,cohort_selection=select,schedule_success=evidence)
        self.assertTrue(out['rpe_success_qualified']);self.assertFalse(out['scientific_R_adopted'])
        evidence['implementation_and_law']=False
        self.assertFalse(r.evaluate_menu(menu,ns,cs,cohort_selection=select,schedule_success=evidence)['rpe_success_qualified'])
        partial=r.evaluate_menu(menu,ns[1:],cs,cohort_selection=select)
        evidence['implementation_and_law']=True;evidence['context']=partial['schedule_success_context']
        self.assertFalse(r.evaluate_menu(menu,ns[1:],cs,cohort_selection=select,schedule_success=evidence)['rpe_success_qualified'])
        evidence['context']['selected_candidate_ids']=['FOREIGN']
        with self.assertRaisesRegex(ValueError,'success evidence'):r.evaluate_menu(menu,ns,cs,cohort_selection=select,schedule_success=evidence)

    def test_metadata_only_entry_and_unknown_shared_uncertainty(self):
        _,menu,ns,cs,select=inputs()
        with patch.object(a,'aggregate_supplied_axes',side_effect=AssertionError('No cost arithmetic in coverage')):
            out=r.coverage(menu,ns,cs,cohort_selection=select)
        self.assertIsNone(out['costs'])
        out=r.evaluate_menu(menu,ns,cs,cohort_selection=select)
        self.assertTrue(out['supplied_cost_evaluation']);self.assertEqual(out['external_N_C_provider_calls'],0)
        u=out['uncertainty'];self.assertIsNone(u['covariance']);self.assertIsNone(u['cross_candidate_covariance'])
        self.assertFalse(u['missing_covariance_is_zero']);self.assertFalse(u['selection_induced_uncertainty_qualified'])
        self.assertTrue(all(v['uncertainty']['shared_calibration_group']=='MOCK_SHARED' for v in u['price_records']))
        self.assertFalse(out['automatic_retry'])


if __name__=='__main__':unittest.main(verbosity=2)
