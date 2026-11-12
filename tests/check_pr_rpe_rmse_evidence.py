"""Mock finite menus only. No molecular schedules, scientific reanalysis or G."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
from trottertracks import pr_rpe_rmse_evidence as E
from check_pr_rpe_rmse_controller import spec, data, m as B
from check_pr_rpe_method_coverage import truncated, qualified_truncated
from check_pr_rpe_rmse_integration import inputs, I


def entry(c, values=None):
    n, prices, cohorts = data(c) if values is None else values
    return dict(configuration=c, N_packet=n, prices=prices,
        cohorts=[dict(candidate_id=cid, axis=axis, cohort=cohort) for (cid, axis), cohort in cohorts.items()])


def four():
    cut = truncated()
    return [entry(B.generate(spec('PF', M=0))), entry(cut, qualified_truncated(cut)),
            entry(B.generate(spec(M=1))), entry(B.generate(spec('FULL_H_RTE', M=1)))]


class EvidenceTests(unittest.TestCase):
    def test_four_methods_ready_evidence_never_costs_or_selects_a_winner(self):
        entries = four(); entries[0]['N_packet']['rounds'][0]['N_X'] = 2**1200 + 3
        before = deepcopy(entries)
        with patch.object(B, 'cost', side_effect=AssertionError('No G')):
            result = E.plan(entries, price_policy='qualified_expectation')
        self.assertEqual(entries, before)
        self.assertEqual(set(result['method_coverage']), set(B.COMPARISON_METHODS))
        self.assertTrue(result['evidence_menu_resolved']); self.assertTrue(result['requested_methods_have_ready_evidence'])
        self.assertEqual(result['price_policy'], 'qualified_expectation')
        self.assertIsNone(result['winner']); self.assertIsNone(result['G']); self.assertFalse(result['comparison_complete'])
        self.assertTrue(all(c['evidence_ready'] for c in result['configurations']))
        for c in result['configurations']:
            for cell in c['rows']:
                self.assertTrue(cell['cost_evaluation_possible']); self.assertEqual(cell['next_work'], [])
                self.assertEqual(cell['configuration_id'], c['configuration_id'])

    def test_missing_axes_and_unexplored_configs_are_explicit(self):
        entries = four(); entries[2]['prices'] = entries[2]['prices'][:-1]
        result = E.plan(entries, price_policy='qualified_expectation', unexplored_scope=['USER_HAS_NOT_FIXED_ADDITIONAL_PF_WIDTHS'])
        self.assertEqual(result['evidence_scope'], 'AVAILABLE_QUALIFIED_SUBSET')
        self.assertFalse(result['evidence_menu_resolved']); self.assertIsNone(result['winner'])
        blocked = result['configurations'][2]
        self.assertFalse(blocked['evidence_ready'])
        self.assertTrue(any(w['owner'] == 'Codex C' for r in blocked['rows'] for w in r['next_work']))
        self.assertTrue(all(not r['cost_evaluation_possible'] for r in blocked['rows']))
        partial = E.plan(entries[:1], price_policy='qualified_expectation')
        self.assertTrue(partial['evidence_menu_resolved'])
        self.assertFalse(partial['requested_methods_have_ready_evidence'])

    def test_full_identity_legacy_record_and_required_source_mismatch(self):
        e = entry(B.generate(spec(M=1)))
        e['prices'][0]['identity']['candidate']['id'] = 'HISTORICAL_PHYSICAL_ID'
        e['source_requirements'] = [dict(m=0, axis='Y', N={'commit': 'OTHER_N_SOURCE'})]
        result = E.plan([e], price_policy='qualified_expectation')['configurations'][0]
        self.assertFalse(result['evidence_ready'])
        self.assertIn('N_REQUIRED_SOURCE_MISMATCH', result['reasons'])
        self.assertEqual(result['unmatched_prices'][0]['candidate_id'], 'HISTORICAL_PHYSICAL_ID')
        self.assertEqual(result['rows'][0]['C']['availability'], 'MISSING')
        old = deepcopy(e); old['configuration']['schema'] = 'PR_RPE_RMSE_V1'
        with self.assertRaisesRegex(ValueError, 'V2'): E.plan([old], price_policy='qualified_expectation')

    def test_local_two_time_packet_keeps_other_rounds_and_sufficient_N_null(self):
        e = entry(B.generate(spec(M=3)))
        n = e['N_packet']; n['status'] = 'NOT_EVALUATED'; n['evidence_kind'] = 'PREFLIGHT_ONLY'
        n['rmse_evidence'].update(rmse_bound_ha=None, covers_state_target_alias_prior_decoder_all_rounds=False)
        for r in n['rounds']:
            r.update(N_X=None, N_Y=None, nominal_N_X=100 if r['m'] in (0, 2) else None)
        e['diagnostic_references'] = [dict(quantity='LOCAL_TWO_TIME_RMSE_DIAGNOSTIC', source={'fixture':'PILOT_ONLY'})]
        out = E.plan([e], price_policy='qualified_expectation')['configurations'][0]
        self.assertEqual(len(out['rows']), 8)
        self.assertTrue(all(r['N']['value_status'] == 'MISSING' for r in out['rows']))
        self.assertTrue(all(not r['error_contract']['all_rounds'] for r in out['rows']))
        self.assertFalse(out['diagnostics_promoted_to_N_C']); self.assertFalse(out['evidence_ready'])
        self.assertTrue(all(r['N_Y'] is None for r in e['N_packet']['rounds']))

    def test_high_probability_N_and_S0a_are_not_N_C(self):
        e = entry(B.generate(spec()))
        e['N_packet']['policy']['family'] = 'HIGH_PROBABILITY'
        e['prices'][0]['quantity'] = 'SINGLE_PATH_STRUCTURAL_PROXY'
        out = E.plan([e], price_policy='qualified_expectation')['configurations'][0]
        self.assertFalse(out['evidence_ready'])
        self.assertIn('N_CONFIGURATION_OR_POLICY_MISMATCH', out['reasons'])
        self.assertIn('C_STRUCTURAL_OR_LOCAL_QUANTITY_IS_NOT_MEAN_COMPILED_PRICE', out['reasons'])

    def test_C1_false_finite_cohort_and_outside_domain_keep_their_meaning(self):
        c = B.generate(spec()); e = entry(c, data(c, model=True))
        e['prices'][0]['model']['qualified_for_candidate'] = False
        e['prices'][1]['price_status'] = 'OUTSIDE_DOMAIN'
        out = E.plan([e], price_policy='qualified_expectation')['configurations'][0]
        self.assertFalse(out['evidence_ready']); self.assertEqual(out['rows'][1]['C']['qualification'], 'OUTSIDE_DOMAIN')
        self.assertFalse(out['rows'][0]['C']['model_qualified_for_candidate'])
        self.assertIsNone(out['rows'][0]['C']['uncertainty']['joint_covariance'])
        e = entry(c, data(c, finite=True))
        self.assertFalse(E.plan([e], price_policy='qualified_expectation')['configurations'][0]['evidence_ready'])
        self.assertTrue(E.plan([e], price_policy='finite_cohort_scenario')['configurations'][0]['evidence_ready'])

    def test_comparison_H_state_sector_RMSE_h_profile_must_match(self):
        base = entry(B.generate(spec(kappa=1)))
        changes = [lambda s:s['binding'].update(hamiltonian='OTHER_H'),
            lambda s:s['binding'].update(state='OTHER_STATE'), lambda s:s['binding'].update(sector='OTHER_SECTOR'),
            lambda s:s['rmse_policy'].update(target_rmse_ha=2), lambda s:s.update(h=2),
            lambda s:s['profile'].update(compiler='OTHER_COMPILER'), lambda s:s['profile'].update(P=1)]
        for change in changes:
            s = spec(kappa=2); change(s)
            with self.assertRaises(ValueError):
                E.plan([base, entry(B.generate(s))], price_policy='qualified_expectation')
        e = entry(B.generate(spec())); e['prices'][0]['units']['rz_count'] = 'LOCAL_EVENT_COUNT'
        self.assertFalse(E.plan([e], price_policy='qualified_expectation')['configurations'][0]['evidence_ready'])

    def test_PR_common_kappa_H6_early_override_and_B3(self):
        c = B.generate(I.a_spec(inputs(M=4, system='H6')))
        result = E.plan([dict(configuration=c)], price_policy='qualified_expectation')['configurations'][0]
        x = [r for r in result['rows'] if r['axis'] == 'X']
        self.assertEqual([r['physical_candidate']['prefix'] for r in x], [5,5,5,5,7])
        self.assertEqual([r['physical_candidate']['R_outer'] for r in x[:4]], [32]*4)
        self.assertTrue(all(r['source_candidate_id'] == r['physical_candidate']['id'] for r in x))
        e = entry(c); e['N_packet']['rounds'][0]['source_candidate_id'] = 'FOREIGN_PHYSICAL_SOURCE'
        out = E.plan([e], price_policy='qualified_expectation')['configurations'][0]
        self.assertFalse(out['evidence_ready']); self.assertIn('N_SOURCE_CANDIDATE_ID_MISMATCH', out['reasons'])
        c['rounds'][4]['law_binding']['kappa']['value'] = 999
        with self.assertRaisesRegex(ValueError, 'mutated'): E.plan([dict(configuration=c)], price_policy='qualified_expectation')
        s = spec('FULL_H_RTE'); s['split']['one_body_policy'] = 'DETERMINISTIC_B3'
        with self.assertRaisesRegex(ValueError, 'B3'): B.generate(s)

    def test_infeasible_does_not_request_unnecessary_C_but_source_mismatch_cannot_prune(self):
        e = entry(B.generate(spec())); e['N_packet']['rmse_evidence']['status'] = 'INFEASIBLE'
        e['prices'] = []
        out = E.plan([e], price_policy='qualified_expectation')['configurations'][0]
        self.assertTrue(out['infeasibility_usable']); self.assertTrue(all(r['next_work'] == [] for r in out['rows']))
        e['source_requirements'] = [dict(m=0,axis='X',N={'fixture':'OTHER_SOURCE'})]
        out = E.plan([e], price_policy='qualified_expectation')['configurations'][0]
        self.assertFalse(out['infeasibility_usable'])
        self.assertTrue(any(w['owner']=='Codex A' for r in out['rows'] for w in r['next_work']))

    def test_explicit_menu_sources_cohorts_and_empty_inputs(self):
        e = entry(B.generate(spec())); e['cohorts'].append(e['cohorts'][0])
        with self.assertRaisesRegex(ValueError, 'cohort'): E.plan([e], price_policy='qualified_expectation')
        e = entry(B.generate(spec())); e['source_requirements'] = [dict(m=99, axis='X')]
        with self.assertRaisesRegex(ValueError, 'source'): E.plan([e], price_policy='qualified_expectation')
        with self.assertRaises(ValueError): E.plan([], price_policy='qualified_expectation')
        s = spec('PF'); s['formula'] = '8th(Morales)'
        with self.assertRaisesRegex(ValueError, 'supported fixed formula'): B.generate(s)
        e = dict(configuration=B.generate(spec()), source_requirements=[dict(m=0,axis='X',N={'fixture':'EXPECTED'})])
        out = E.plan([e], price_policy='qualified_expectation')['configurations'][0]
        self.assertEqual(out['rows'][0]['N']['source_constraint'], 'MISSING')
        self.assertIn('N_REQUIRED_SOURCE_MISSING', out['reasons'])

    def test_full_RTE_total_events_do_not_erase_source_grouping(self):
        s = spec('FULL_H_RTE', M=0); s['event_mapping']['kind'] = 'MOCK_GIVEN_LAYOUT'
        def config(q, r):
            def supplied(s, row):
                return dict(kind=s['event_mapping']['kind'], source=s['event_mapping']['source'],
                    kappa=s['kappa'], R_outer=None, event_count=q*r,
                    physical_layout=dict(legacy_q=q, legacy_R=r, priced_equivalence_claim=False))
            return B.generate(s, event_mapper=supplied)
        c1 = config(1, 6); c2 = config(2, 3)
        self.assertNotEqual(c1['id'], c2['id'])
        out = E.plan([entry(c1), entry(c2)], price_policy='qualified_expectation')
        self.assertEqual([r['rows'][0]['physical_schedule_mapping']['physical_layout']['legacy_q']
                          for r in out['configurations']], [1, 2])
        self.assertTrue(all(r['rows'][0]['physical_schedule_mapping']['event_count'] == 6 for r in out['configurations']))
        wrong = entry(c2); wrong['N_packet'] = data(c1)[0]
        self.assertFalse(E.plan([wrong], price_policy='qualified_expectation')['configurations'][0]['evidence_ready'])

    def test_metadata_never_calls_science_provider_cost_or_G_selection(self):
        entries = four(); original = B.select
        def metadata_select(configs, evaluations, **kw):
            self.assertTrue(all(v['cost'] is None for v in evaluations))
            return original(configs, evaluations, **kw)
        with (patch.object(B, 'cost', side_effect=AssertionError('No G')),
              patch.object(B.p, 'run_missing', side_effect=AssertionError('No provider')),
              patch.object(I.A, 'evaluate', side_effect=AssertionError('No numerical A')),
              patch.object(B, 'select', side_effect=metadata_select)):
            self.assertEqual(E.plan(entries, price_policy='qualified_expectation')['automatic_science_calls'], 0)
        subprocess.run([sys.executable, '-B', '-c', 'import sys; from trottertracks.pr_rpe_rmse_evidence import plan; '
            "assert not any(n in sys.modules for n in ('numpy','scipy','qiskit'))"],
            cwd=ROOT, env=dict(os.environ, PYTHONPATH='src'), check=True)

    def test_cli_requires_explicit_menu_and_writes_new_output_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'menu.json'; out = root/'plan.json'
            source.write_text(json.dumps(dict(entries=four(), price_policy='qualified_expectation',
                required_methods=list(B.COMPARISON_METHODS), unexplored_scope='SYNTHETIC_ONLY')))
            command=[sys.executable,'-B','scripts/plan_pr_rpe_rmse_evidence.py','--menu',str(source),'--out',str(out)]
            subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result=json.loads(out.read_text()); self.assertIsNone(result['G']); self.assertIsNone(result['winner'])
            retry=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertNotEqual(retry.returncode,0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
