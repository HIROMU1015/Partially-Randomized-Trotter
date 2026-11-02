"""Invented H2-shaped summary fixtures only; never read actual D/A statistics."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h2_s_stats_contract as s
import h2_s_stats_numeric as numeric
import run_h2_s_stats as runner
import prepare_h2_s_stats as prep
import audit_h2_s_stats_completed as audit
c=s.c


def fixture():
    cells={x['id']:x for x in s.main.registrations()['A']};endpoints=[]
    for i,(epsilon,cid,pf) in enumerate(s.TARGETS):
        endpoints.append(dict(epsilon=epsilon,cell=deepcopy(cells[cid]),
            PR=dict(cell_id=cid,epsilon=epsilon,eligible=True,cohort='EXPLORATION_N8',n=8,
                G_RZ_exact='80' if i==0 else '150',G_RZ_SE=1 if i==0 else .5),
            PF=dict(cell_id=pf,epsilon=epsilon,eligible=True,G_RZ_exact='100' if i==0 else '200'),
            saved_PR_to_PF_ratio=.8 if i==0 else .75,saved_rare_counts=dict(S0_ONLY=8)))
    return dict(binding=deepcopy(s.main.BINDING),endpoints=endpoints,old_FP_or_legacy_pooled=False)


def upstream_fixture():
    summary=fixture()
    a=dict(stage='A',SOURCE=s.A_SOURCE,status='COMPLETE_MANDATORY_STOP',binding=deepcopy(s.main.BINDING),
        accounting=dict(rows=[deepcopy(e['PR']) for e in summary['endpoints']],winners={},comparisons=[]))
    d=dict(stage='D',SOURCE=s.D_SOURCE,status='COMPLETE_MANDATORY_STOP',binding=deepcopy(s.main.BINDING),
        accounting=dict(rows=[deepcopy(e['PF']) for e in summary['endpoints']],winners={}))
    for i,e in enumerate(summary['endpoints']):
        a['accounting']['winners'][str(e['epsilon'])]=a['accounting']['rows'][i]
        d['accounting']['winners'][str(e['epsilon'])]=d['accounting']['rows'][i]
        a['accounting']['comparisons'].append(dict(epsilon=e['epsilon'],PR=a['accounting']['rows'][i],
            PF=d['accounting']['rows'][i],PR_to_PF_ratio=e['saved_PR_to_PF_ratio'],status='FINITE_SAMPLE_POINT_COMPARISON'))
    rare=dict(winner_stratum_counts={e['cell']['id']:e['saved_rare_counts'] for e in summary['endpoints']})
    return a,d,rare


class NumericReuse(unittest.TestCase):
    def test_H2_namespace_and_original_shared_globals_unchanged(self):
        before=(numeric.shared.s,numeric.shared.c);result=numeric.analyze(fixture())
        self.assertEqual((numeric.shared.s,numeric.shared.c),before)
        self.assertEqual(result['plan']['binding']['rank'],3)
        self.assertEqual(result['plan']['scope'],s.SCOPE)
        self.assertEqual(len(result['normal_scenarios']),24)
        self.assertEqual(len(result['sensitivity']),12)
        self.assertEqual(len(result['reference_directions']),2)

    def test_known_normal_ceiling_and_reference(self):
        result=numeric.analyze(fixture());cid=s.TARGETS[0][1]
        rows=[v for v in result['normal_scenarios'] if v['cell_id']==cid and v['relative_gap']==.02]
        self.assertEqual([v['n'] for v in rows],[16,63,252,1005])
        self.assertEqual(result['recommended']['n_fixed'],{s.TARGETS[0][1]:252,s.TARGETS[1][1]:16})
        self.assertIsNone(result['plan']['n_cap'])

    def test_H2_P_proposal_counts_are_not_H4_counts(self):
        v=numeric.analyze(fixture())['recommended']['proposed_P_calls']
        self.assertEqual({k:v[k] for k in ('trajectory','wrapper_compile','occurrence','RTE_event','seed_slots')},
            dict(trajectory=268,wrapper_compile=536,occurrence=316,RTE_event=632,seed_slots=332))
        self.assertFalse(v['P_executed'])

    def test_no_input_mutation_or_rare_tail_filling(self):
        x=fixture();before=deepcopy(x);result=numeric.analyze(x);self.assertEqual(x,before)
        for v in result['rare_information']:
            self.assertEqual(v['saved_counts'],dict(S0_ONLY=8));self.assertIsNone(v['law_probabilities'])
            self.assertIsNone(v['n_visibility']);self.assertIsNone(v['unseen_cost'])
        self.assertFalse(result['actual_power_certified']);self.assertFalse(result['population_mean_certified'])

    def test_exact_fraction_direction_tie_and_opposite(self):
        x=fixture();a,b=x['endpoints'];a['PR']['G_RZ_exact']='99999999999999999999/1000000000000000000'
        a['saved_PR_to_PF_ratio']=.999;b['PR']['G_RZ_exact']='201';b['saved_PR_to_PF_ratio']=1.005
        rows=numeric.analyze(x)['reference_directions']
        self.assertEqual([v['expected_direction'] for v in rows],[1,-1])
        self.assertEqual(rows[0]['PF_minus_PR_exact'],'1/1000000000000000000')
        a['PR']['G_RZ_exact']='100';a['saved_PR_to_PF_ratio']=1
        self.assertEqual(numeric.analyze(x)['reference_directions'][0]['expected_direction'],0)

    def test_bad_SE_n_FP_cohort_and_foreign_binding_rejected(self):
        for kind in ('SE0','SEnan','n','FP','binding','pooled'):
            x=fixture()
            if kind=='SE0':x['endpoints'][0]['PR']['G_RZ_SE']=0
            elif kind=='SEnan':x['endpoints'][0]['PR']['G_RZ_SE']=float('nan')
            elif kind=='n':x['endpoints'][0]['PR']['n']=32
            elif kind=='FP':x['endpoints'][0]['PR']['cohort']='FP_NATIVE_DIAGNOSTIC_N1'
            elif kind=='binding':x['binding']['sector_dimension']=6
            else:x['old_FP_or_legacy_pooled']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):numeric.analyze(x)

    def test_missing_or_changed_endpoint_rejected(self):
        for kind in ('missing','changed'):
            x=fixture()
            if kind=='missing':x['endpoints'].pop()
            else:x['endpoints'][0]['epsilon']=.01
            with self.assertRaisesRegex(ValueError,'FIXED_PRIMARY'):numeric.analyze(x)

    def test_zero_SE_floor_not_used_and_minimum_two_for_tiny_positive_SE(self):
        x=fixture()
        for e in x['endpoints']:e['PR']['G_RZ_SE']=1e-12
        self.assertTrue(all(v['n']==2 for v in numeric.analyze(x)['normal_scenarios']))


class PreparationBoundary(unittest.TestCase):
    def test_selected_summary_only_copies_saved_rows_counts_and_ratio(self):
        a,d,rare=upstream_fixture();before=deepcopy((a,d,rare))
        with (patch.object(c,'normal_n_formula',side_effect=AssertionError('NUMERIC_CALL')),
              patch('math.sqrt',side_effect=AssertionError('SD_CALCULATION'))):
            summary=s.selected_summary(a,d,rare)
        self.assertEqual((a,d,rare),before);self.assertEqual(summary['endpoints'][0]['saved_rare_counts'],{'S0_ONLY':8})
        self.assertNotIn('saved_point_direction',summary['endpoints'][0])

    def test_wrong_upstream_SOURCE_and_sector_rejected(self):
        for mode in ('SOURCE','sector'):
            a,d,rare=upstream_fixture()
            if mode=='SOURCE':d['SOURCE']=s.A_SOURCE
            else:a['binding']['sector_dimension']=6
            with self.assertRaisesRegex(ValueError,'ACCEPTED_H2'):s.selected_summary(a,d,rare)

    def test_bad_saved_candidate_or_PF_rejected(self):
        for mode in ('PR','PF'):
            a,d,rare=upstream_fixture()
            target=a['accounting']['rows'][0] if mode=='PR' else d['accounting']['rows'][0]
            target['cell_id']='foreign'
            with self.assertRaisesRegex(ValueError,'FIXED_SAVED'):s.selected_summary(a,d,rare)

    def test_saved_rare_counts_must_cover_n8(self):
        a,d,rare=upstream_fixture();rare['winner_stratum_counts'][s.TARGETS[0][1]]={'S0_ONLY':7}
        with self.assertRaisesRegex(ValueError,'RARE_COUNTS'):s.selected_summary(a,d,rare)

    def test_preparer_contains_no_numerical_dispatch(self):
        tree=ast.parse((s.ROOT/'scripts/prepare_h2_s_stats.py').read_text())
        calls={v.func.attr if isinstance(v.func,ast.Attribute) else v.func.id for v in ast.walk(tree)
            if isinstance(v,ast.Call) and isinstance(v.func,(ast.Name,ast.Attribute))}
        self.assertTrue(calls.isdisjoint({'analyze','normal_n_formula','inv_cdf','sqrt','evaluate_pairs','sample','transpile'}))

    def test_invalid_grant_precedes_input_access(self):
        with patch.object(s,'verify_bytes',side_effect=AssertionError('INPUT_READ')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_H2_S_STATS_APPROVAL_REQUIRED'):
                s.authorize({},dict(approved_by_user=False,runtime_authorization=False))

    def test_foreign_stage_grant_rejected(self):
        for stage in ('A','P'):
            with self.assertRaisesRegex(ValueError,'H2_S_STATS_ONLY'):
                s.authorize(dict(stage='S_STATS'),dict(scope=s.SCOPE,stage=stage,approved_by_user=True,runtime_authorization=True))

    def test_pending_contract_cannot_execute(self):
        with self.assertRaisesRegex(ValueError,'SEALED_STATS_EXECUTION_REQUIRED'):
            s.authorize(dict(stage='S_STATS',status='SEALED_PENDING_ACTIVATION'),dict(scope=s.SCOPE,stage='S_STATS',
                approved_by_user=True,runtime_authorization=True))


class OneShotAndAudit(unittest.TestCase):
    def setUp(self):
        base=s.PRIVATE/'artificial_s_stats';base.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.tmp.name)

    def tearDown(self):self.tmp.cleanup()

    def test_real_CLI_invalid_grant_both_entries(self):
        spec=dict(stage='S_STATS',SOURCE='a'*40,attempt=str(self.root/'uncreated'))
        c.save(self.root/'spec.json',spec);c.save(self.root/'grant.json',s.invalid_grant(spec))
        rows=prep.refusal(self.root/'spec.json',self.root/'grant.json',self.root)
        self.assertEqual(len(rows),2);self.assertFalse((self.root/'uncreated').exists())

    def test_artificial_one_pass_and_no_retry(self):
        path=self.root/'summary.json';c.save(path,fixture())
        spec=dict(SOURCE='a'*40,output=str(self.root/'output'),lock=str(self.root/'ONE_SHOT.json'),summary=str(path))
        with (patch.object(s,'authorize',return_value=fixture()),patch.object(s,'verify_bytes'),
              patch.object(numeric,'analyze',wraps=numeric.analyze) as function):
            value=runner.execute(spec,{})
            with self.assertRaises(FileExistsError):runner.execute(spec,{})
            self.assertEqual(function.call_count,1)
        self.assertEqual(value['actual_calls']['normal_scenarios'],24)
        self.assertTrue(c.read(self.root/'output/calls_ledger.json')['nonrefundable'])

    def test_artificial_failure_keeps_claim_ledger_STOP(self):
        spec=dict(SOURCE='a'*40,output=str(self.root/'output'),lock=str(self.root/'ONE_SHOT.json'))
        with patch.object(s,'authorize',return_value=fixture()),patch.object(numeric,'analyze',side_effect=ValueError('ARTIFICIAL')):
            with self.assertRaisesRegex(ValueError,'ARTIFICIAL'):runner.execute(spec,{})
        self.assertTrue((self.root/'ONE_SHOT.json').exists());self.assertTrue((self.root/'output/STOP.json').exists())
        self.assertTrue((self.root/'output/calls_ledger.json').exists());self.assertFalse((self.root/'output/result.json').exists())

    def test_auditor_does_not_redispatch_formula_or_quantile(self):
        value=numeric.analyze(fixture())
        with (patch.object(c,'normal_n_formula',side_effect=AssertionError('N_REPEATED')),
              patch('statistics.NormalDist.inv_cdf',side_effect=AssertionError('QUANTILE_REPEATED'))):
            audit.check_result(value,fixture())

    def test_auditor_rejects_bad_proposed_budget_and_reference(self):
        for mode in ('budget','reference','rare'):
            value=numeric.analyze(fixture())
            if mode=='budget':value['sensitivity'][0]['proposed_P_calls']['wrapper_compile']+=1
            elif mode=='reference':value['recommended']=value['sensitivity'][0]
            else:value['rare_information'][0]['unseen_cost']=0
            with self.subTest(mode=mode),self.assertRaises(ValueError):audit.check_result(value,fixture())


if __name__=='__main__':unittest.main()
