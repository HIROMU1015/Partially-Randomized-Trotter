"""Artificial summaries only. Never read saved D/A means, costs or raw samples."""
import ast
from copy import deepcopy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h4_tol_s_stats_contract as s
import h4_tol_s_stats_numeric as n
import run_h4_tol_s_stats as runner
import prepare_h4_tol_s_stats as prep
c=s.c


def fixture():
    cells={x['id']:x for x in s.main.registrations()['A']};endpoints=[]
    for i,(eps,cid,pf) in enumerate(s.TARGETS):
        endpoints.append(dict(epsilon=eps,cell=deepcopy(cells[cid]),
            PR=dict(cell_id=cid,cohort='EXPLORATION_N8',n=8,G_RZ_exact=str(80 if i==0 else 150),G_RZ_SE=1 if i==0 else .5),
            PF=dict(cell_id=pf,G_RZ_exact=str(100 if i==0 else 200)),saved_point_direction=1,
            saved_rare_counts=dict(S0_ONLY=8,S2_SINGLE=0,S2_MULTI=0,S4_PRESENT=0)))
    return dict(binding=deepcopy(s.main.BINDING),endpoints=endpoints,old_FP_or_legacy_pooled=False)


class NumericDesign(unittest.TestCase):
    def test_fixed_two_family_and_twenty_four_scenarios(self):
        result=n.analyze(fixture())
        self.assertEqual(len(result['normal_scenarios']),24)
        self.assertEqual(len(result['sensitivity']),12)
        self.assertEqual(len(result['reference_directions']),2)
        self.assertAlmostEqual(result['quantiles']['z_one_sided_Bonferroni'],1.959963984540054)
        self.assertAlmostEqual(result['quantiles']['z_power'],.8416212335729144)

    def test_known_integer_ceiling_and_fixed_reference(self):
        result=n.analyze(fixture());cid=s.TARGETS[0][1]
        rows=[r for r in result['normal_scenarios'] if r['cell_id']==cid and r['relative_gap']==.02]
        self.assertEqual([r['n'] for r in rows],[16,63,252,1005])
        self.assertEqual(result['recommended']['SD_inflation'],4)
        self.assertEqual(result['recommended']['relative_gap'],.02)
        self.assertEqual(result['recommended']['n_fixed'],{s.TARGETS[0][1]:252,s.TARGETS[1][1]:16})

    def test_P_budget_pairs_each_sample_without_double_count(self):
        budget=n.analyze(fixture())['recommended']['proposed_P_calls']
        self.assertEqual({k:budget[k] for k in ('trajectory','wrapper_compile','occurrence','RTE_event','seed_slots')},
            dict(trajectory=268,wrapper_compile=536,occurrence=380,RTE_event=1264,seed_slots=396))
        self.assertEqual(budget['control_probe_maximum'],8);self.assertFalse(budget['P_executed'])

    def test_exact_direction_tie_and_opposite_are_retained(self):
        summary=fixture();a,b=summary['endpoints']
        a['PR']['G_RZ_exact']='99999999999999999999/1000000000000000000'
        b['PR']['G_RZ_exact']='201';b['saved_point_direction']=-1
        rows=n.analyze(summary)['reference_directions']
        self.assertEqual([r['expected_direction'] for r in rows],[1,-1])
        self.assertEqual(rows[0]['PF_minus_PR_exact'],'1/1000000000000000000')
        a['PR']['G_RZ_exact']='100';a['saved_point_direction']=0
        self.assertEqual(n.analyze(summary)['reference_directions'][0]['expected_direction'],0)

    def test_unseen_cost_probability_and_visibility_are_not_invented(self):
        summary=fixture();before=deepcopy(summary);result=n.analyze(summary)
        self.assertEqual(summary,before)
        for rare in result['rare_information']:
            self.assertIsNone(rare['law_probabilities']);self.assertIsNone(rare['n_visibility'])
            self.assertIsNone(rare['unseen_cost']);self.assertFalse(rare['unobserved_strata_zero_filled'])
        self.assertFalse(result['population_mean_certified']);self.assertFalse(result['actual_power_certified'])
        self.assertFalse(result['CI_or_p_values_computed'])

    def test_n_cap_none_and_minimum_two(self):
        summary=fixture()
        for ep in summary['endpoints']:ep['PR']['G_RZ_SE']=1e-9
        result=n.analyze(summary)
        self.assertIsNone(result['plan']['n_cap'])
        self.assertTrue(all(r['n']==2 for r in result['normal_scenarios']))

    def test_missing_endpoint_or_changed_target_rejected(self):
        for mode in ('missing','changed','duplicate'):
            summary=fixture()
            if mode=='missing':summary['endpoints'].pop()
            elif mode=='changed':summary['endpoints'][0]['epsilon']=.01
            else:summary['endpoints'][1]=deepcopy(summary['endpoints'][0])
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'FIXED_PRIMARY'):n.analyze(summary)

    def test_legacy_rank_FP_and_pooled_cohort_rejected(self):
        for mode in ('rank','FP','pooled','n'):
            summary=fixture()
            if mode=='rank':summary['binding']['rank']=12
            elif mode=='FP':summary['endpoints'][0]['PR']['cohort']='SAVED_FP_PANEL_N32_ANCHOR'
            elif mode=='n':summary['endpoints'][0]['PR']['n']=32
            else:summary['old_FP_or_legacy_pooled']=True
            with self.subTest(mode=mode),self.assertRaises(ValueError):n.analyze(summary)

    def test_nonfinite_or_zero_SE_rejected(self):
        for value in (0,-1,float('nan'),float('inf')):
            summary=fixture();summary['endpoints'][0]['PR']['G_RZ_SE']=value
            with self.subTest(value=value),self.assertRaises(ValueError):n.analyze(summary)

    def test_changed_saved_point_direction_rejected(self):
        summary=fixture();summary['endpoints'][0]['saved_point_direction']=-1
        with self.assertRaisesRegex(ValueError,'FIXED_EXPLORATION_DIRECTION'):n.analyze(summary)


class Boundary(unittest.TestCase):
    def test_no_numeric_entry_in_preparer(self):
        tree=ast.parse((s.ROOT/'scripts/prepare_h4_tol_s_stats.py').read_text())
        names={node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id
            for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,(ast.Name,ast.Attribute))}
        self.assertTrue(names.isdisjoint({'analyze','normal_n_formula','inv_cdf','evaluate_pairs','sample','transpile','bootstrap'}))

    def test_invalid_grant_precedes_input_read_and_numerics(self):
        with patch.object(s,'verify_bytes',side_effect=AssertionError('INPUT_READ')):
            with self.assertRaisesRegex(ValueError,'SEPARATE_EXPLICIT_S_STATS_APPROVAL_REQUIRED'):
                s.authorize({},dict(approved_by_user=False,runtime_authorization=False))

    def test_P_or_A_grant_never_authorizes_stats(self):
        for stage in ('P','A','D'):
            with self.subTest(stage=stage),self.assertRaisesRegex(ValueError,'DF10_S_STATS_ONLY'):
                s.authorize(dict(stage='S_STATS'),dict(stage=stage,scope=s.SCOPE,approved_by_user=True,runtime_authorization=True))

    def test_preparation_receipt_cannot_activate_pending_contract(self):
        with self.assertRaisesRegex(ValueError,'PENDING_EXECUTION_APPROVAL'):
            s.authorize(dict(stage='S_STATS',status='TECHNICALLY_SEALED_PENDING_EXPLICIT_STATS_APPROVAL'),
                dict(stage='S_STATS',scope=s.SCOPE,approved_by_user=True,runtime_authorization=True))

    def test_modified_plan_or_incomplete_source_refused(self):
        with self.assertRaisesRegex(ValueError,'FIXED_STATS_PLAN'):s.verify_bytes(dict(plan={}))
        with self.assertRaisesRegex(ValueError,'COMPLETE_STATS_SOURCE_CLOSURE'):
            s.verify_bytes(dict(plan=s.plan(),source_hashes={}))

    def test_input_corruption_rejected_before_further_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'summary';path.write_text('{}')
            spec=dict(plan=s.plan(),source_hashes={},inputs=[dict(role='synthetic',path=str(path),sha256='0'*64)])
            with patch.object(s,'source_closure',return_value={}):
                with self.assertRaisesRegex(ValueError,'STATS_INPUT_CHANGED'):s.verify_bytes(spec)

    def test_rebound_spec_digest_rejected(self):
        spec=dict(stage='S_STATS',status='SEALED_FOR_EXECUTION',pending=[],SOURCE='a'*40)
        grant=dict(stage='S_STATS',scope=s.SCOPE,approved_by_user=True,runtime_authorization=True,
            one_shot=True,automatic_retry=False,next_stage_authorized=False,SOURCE='a'*40,spec_digest='changed')
        with self.assertRaisesRegex(ValueError,'STATS_CONTRACT_BINDING'):s.authorize(spec,grant)

    def test_missing_fresh_admission_rejected_before_numeric_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'receipt').write_text('artificial approval')
            spec=dict(stage='S_STATS',status='SEALED_FOR_EXECUTION',pending=[],SOURCE='a'*40,
                output=str(p/'unused/output'),lock=str(p/'unused/lock'),fresh_resources=dict(admitted=False))
            grant=dict(stage='S_STATS',scope=s.SCOPE,approved_by_user=True,runtime_authorization=True,
                one_shot=True,automatic_retry=False,next_stage_authorized=False,SOURCE='a'*40,
                spec_digest=c.digest(spec),approval_receipt=str(p/'receipt'),approval_receipt_sha256=c.sha(p/'receipt'))
            with patch.object(s,'verify_bytes',side_effect=AssertionError('NUMERIC_INPUT')):
                with self.assertRaisesRegex(ValueError,'FRESH_STATS_RESOURCE_RECEIPT'):s.authorize(spec,grant)

    def test_actual_direct_and_witness_refusal_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);spec=dict(stage='S_STATS',SOURCE='a'*40,output=str(p/'unused/output'),lock=str(p/'unused/ONE_SHOT'))
            c.save(p/'spec',spec);c.save(p/'grant',s.invalid_grant(spec))
            rows=prep.refusal(p/'spec',p/'grant',p)
            self.assertEqual(len(rows),2);self.assertTrue(all(r['refused_before_numeric_import'] for r in rows))
            self.assertFalse((p/'unused').exists())

    def test_artificial_one_shot_runs_once_and_retains_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);c.save(p/'summary',fixture())
            spec=dict(SOURCE='a'*40,output=str(p/'output'),lock=str(p/'ONE_SHOT'),summary=str(p/'summary'))
            with patch.object(s,'authorize',return_value=fixture()),patch.object(s,'verify_bytes'),patch.object(n,'analyze',wraps=n.analyze) as numeric:
                result=runner.execute(spec,{})
                with self.assertRaises(FileExistsError):runner.execute(spec,{})
                self.assertEqual(numeric.call_count,1)
            self.assertEqual(result['actual_calls']['normal_scenarios'],24)
            self.assertTrue(c.read(p/'output/calls_ledger.json')['nonrefundable'])
            self.assertEqual(c.read(p/'output/terminal.json')['status'],'COMPLETE_MANDATORY_STOP')

    def test_artificial_failure_preserves_STOP_and_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);spec=dict(SOURCE='a'*40,output=str(p/'output'),lock=str(p/'ONE_SHOT'))
            with patch.object(s,'authorize',return_value=fixture()),patch.object(n,'analyze',side_effect=ValueError('ARTIFICIAL')):
                with self.assertRaisesRegex(ValueError,'ARTIFICIAL'):runner.execute(spec,{})
            self.assertTrue((p/'ONE_SHOT').exists());self.assertTrue((p/'output/calls_ledger.json').exists())
            self.assertFalse(c.read(p/'output/STOP.json')['automatic_retry'])
            self.assertFalse((p/'output/result.json').exists())


if __name__=='__main__':unittest.main()
