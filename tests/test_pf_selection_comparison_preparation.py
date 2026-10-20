"""Artificial arithmetic and literal/structural checks only; no real ranking."""
import ast
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'))
sys.path.insert(0,str(ROOT/'scripts'))
import pf_selection_comparison_plan as p
import prepare_pf_selection_comparison as preparation


class SyntheticTests(unittest.TestCase):
    def fixture(self):
        rows={'A':dict(eligible=True,shots={'cosine':2,'sine':3},RZ={'cosine':'10','sine':'20'},q=1),
              'B':dict(eligible=True,shots={'cosine':2,'sine':3},RZ={'cosine':'20','sine':'20'},q=2)}
        return dict(kind='ARTIFICIAL',baseline=rows,direct=deepcopy(rows),saved_PR_total='90')

    def test_real_execution_refused(self):
        for value in ({},{'kind':'SAVED_REAL_VALUES'}):
            with self.assertRaises(PermissionError):p.synthetic_compare(value)
        with self.assertRaises(PermissionError):p.reject_production()

    def test_fixed_axis_weighting_and_no_mutation(self):
        data=self.fixture();before=deepcopy(data);result=p.synthetic_compare(data)
        self.assertEqual(data,before)
        self.assertEqual(result['baseline']['total'],'80')
        self.assertEqual(result['PR_over_PF_before'],'9/8')
        self.assertFalse(result['best_candidate_changed'])

    def test_candidate_q_and_PR_PF_reversal(self):
        data=self.fixture();data['direct']['A']['RZ']={'cosine':'30','sine':'30'}
        result=p.synthetic_compare(data)
        self.assertEqual(result['baseline']['best_meta']['q'],1)
        self.assertEqual(result['direct']['best_meta']['q'],2)
        self.assertTrue(result['best_candidate_changed'])
        self.assertEqual(result['PR_PF_relation_before'],'PF_LOWER')
        self.assertEqual(result['PR_PF_relation_after'],'PR_LOWER')
        self.assertEqual(result['pairwise_strict_inversions'],[['A','B']])

    def test_exact_tie_is_not_strict_inversion(self):
        data=self.fixture();data['direct']['A']['RZ']=deepcopy(data['direct']['B']['RZ'])
        result=p.synthetic_compare(data)
        self.assertEqual(result['direct']['best'],'A')
        self.assertEqual(result['direct']['ties'],['A','B'])
        self.assertEqual(result['pairwise_strict_inversions'],[])
        self.assertEqual(result['pairwise_tie_changes'],[['A','B']])

    def test_identical_subset_required(self):
        data=self.fixture();del data['direct']['B']
        with self.assertRaises(ValueError):p.synthetic_compare(data)

    def test_saved_shots_and_eligibility_cannot_change(self):
        for field,value in [('shots',{'cosine':1,'sine':3}),('eligible',False)]:
            data=self.fixture();data['direct']['A'][field]=value
            with self.assertRaises(ValueError):p.synthetic_compare(data)

    def test_missing_and_unresolved_never_become_zero(self):
        data=self.fixture();data['direct']['A']['RZ']['sine']=None
        with self.assertRaises(ValueError):p.synthetic_compare(data)
        data=self.fixture()
        for arm in ('baseline','direct'):data[arm]['A']['eligible']=None
        with self.assertRaises(ValueError):p.synthetic_compare(data)

    def test_ineligible_not_ranked_and_no_PR_cost_missing_is_null(self):
        data=self.fixture()
        for arm in ('baseline','direct'):data[arm]['A']['eligible']=False
        data['saved_PR_total']=None
        result=p.synthetic_compare(data)
        self.assertEqual(result['baseline']['best'],'B')
        self.assertIsNone(result['PR_over_PF_before'])

    def test_coverage_is_presence_only_and_distinguishes_empty(self):
        row=dict(candidate_id='A',eligible=True,saved_shots={'cosine':1,'sine':2},price_coverage='COMPLETE')
        self.assertEqual(p.coverage([row],['A'])['status'],'CONDITIONAL_INPUTS_READY')
        self.assertEqual(p.coverage([row],[])['status'],'NO_ELIGIBLE_CANDIDATE')
        row['price_coverage']='MISSING';self.assertEqual(p.coverage([row],['A'])['status'],'BLOCKED_DIRECT_PRICE_MISSING')
        row['eligible']=None;self.assertEqual(p.coverage([row],['A'])['status'],'BLOCKED_SAVED_ELIGIBILITY_UNRESOLVED')

    def test_preparation_run_cli_refused_before_project_inputs(self):
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/prepare_pf_selection_comparison.py'),'--run'],capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn(b'PREPARATION_ONLY_NO_REAL_SHOT_COST_OR_RANKING_EXECUTION',result.stderr)


class StructuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=preparation.OUT
        cls.rows=json.loads((cls.out/'saved_condition_rows.json').read_text())['rows']
        cls.units=json.loads((cls.out/'axis_price_lineage.json').read_text())['rows']
        cls.coverage=json.loads((cls.out/'condition_coverage.json').read_text())['rows']

    def test_saved_row_domains_and_confirmation_cohorts(self):
        self.assertEqual(len(self.rows),6680)
        self.assertEqual(len({r['row_id'] for r in self.rows}),6680)
        pr=[r for r in self.rows if r['kind']=='PR']
        self.assertEqual(len(pr),280)
        self.assertTrue(all(r['paired_samples']==32 for r in pr if r['eligible'] is True))
        self.assertFalse(any('EXPLORATION' in r['cohort'] for r in self.rows))

    def test_new450_extra8_and_existing_prices_remain_distinct(self):
        self.assertEqual(len(self.units),640)
        self.assertEqual(sum(u['ALL_SAVED458']=='REPLACE_NEW450' for u in self.units),450)
        self.assertEqual(sum(u['ALL_SAVED458']=='REPLACE_EXISTING_EXTRA8' for u in self.units),8)
        self.assertEqual(sum(u['ALL_SAVED458']=='RETAIN_OLD_MEASURED' for u in self.units),102)
        missing=[u for u in self.units if u['ALL_SAVED458']=='MISSING_BASELINE_PRICE']
        self.assertEqual(len(missing),80)
        self.assertTrue(all(u['baseline_RZ_literal'] is None for u in missing))

    def test_conditional_counts_not_new_numeric_results(self):
        self.assertEqual(len(self.coverage),300)
        full=[r for r in self.coverage if r['strategy']=='ALL_SAVED458']
        self.assertEqual(sum(r['status']=='CONDITIONAL_INPUTS_READY' for r in full),70)
        self.assertEqual(sum(r['status']=='BLOCKED_SAVED_ELIGIBILITY_UNRESOLVED' for r in full),30)
        self.assertTrue(all(not r['is_new_numeric_result'] for r in self.coverage))
        self.assertTrue(all(r['domain_kind']=='LIMITED_MATCHED_SUBSET' for r in self.coverage if r['strategy']=='MATCHED_NEW450_SUBSET'))

    def test_eligible_saved_shots_transcribed_from_policy(self):
        for r in self.rows:
            if r['eligible'] is True:
                ac=r['saved_bias_and_shot_policy']
                self.assertEqual(r['saved_shots'],{'cosine':ac['axes']['real']['shots'],'sine':ac['axes']['imag']['shots']})

    def test_old_pending48_and_initial_final_gate_flags_preserved(self):
        self.assertEqual(sum(u['old_profile_pending_48'] for u in self.units),48)
        selected=[u['new_direct'] for u in self.units if u['ALL_SAVED458']=='REPLACE_NEW450']
        self.assertEqual(sum(r['original_global_final_gate_missing'] for r in selected),428)

    def test_extra8_old_schema_prices_and_source_are_real_saved_metadata(self):
        rows=json.loads((self.out/'extra8_existing_receipts.json').read_text())['rows']
        self.assertEqual(len(rows),8)
        for r in rows:
            self.assertTrue(r['direct_RZ_literal'].isdigit())
            self.assertGreater(int(r['direct_RZ_literal']),0)
            self.assertEqual(len(r['direct_SOURCE']),40)
        self.assertTrue(all(r['direct_RZ_literal']=='14492' for r in rows if r['evidence_source_id'].startswith('H6_P7_')))

    def test_contract_invalid_and_templates_have_no_real_result_rows(self):
        contract=json.loads((self.out/'execution_contract_DRAFT.json').read_text())
        self.assertFalse(contract['scientific_execution_authorized']);self.assertIsNone(contract['active_grant'])
        grant=json.loads((self.out/'grant_INVALID.json').read_text());self.assertFalse(grant['approved'])
        for name in ('candidate_totals','selection_comparison','rank_changes'):
            self.assertEqual(len(list(csv.DictReader(io.StringIO((self.out/(name+'_TEMPLATE.csv')).read_text())))),0)

    def test_builder_does_not_call_synthetic_or_legacy_accounting(self):
        tree=ast.parse((ROOT/'scripts/prepare_pf_selection_comparison.py').read_text())
        forbidden={'synthetic_compare','minimum','min','evaluate','reaccount','axis_errors','diagnose','Fraction'}
        calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
        self.assertFalse(calls&forbidden)


if __name__=='__main__':unittest.main()
