"""Fixed24 metadata, artificial arithmetic and mock execution. No real log B."""
from copy import deepcopy
import json,math,os,subprocess,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pr_r_normalization_phase0 as phase


def artificial(packet):
    packet=deepcopy(packet)
    for system in packet['tails']:
        packet['tails'][system]['lambda_R']=.5
        packet['tails'][system]['source']=dict(fixture='ARTIFICIAL_TAIL_NOT_MOLECULAR')
    return packet


def mocked_score(root,proposal,tail,**kwargs):
    assert tail['source']==dict(fixture='ARTIFICIAL_TAIL_NOT_MOLECULAR')
    assert kwargs==dict(shot_inflation_budget=None,budget_source=None)
    rows=deepcopy(proposal['rows'])
    for row in rows:row['normalization'].update(status='NORMALIZATION_SCORED_NOT_N_QUALIFIED',log_B=.25,optimistic_shot_inflation_log=.5,one_ulp_accumulation_log_sensitivity=1e-12)
    return dict(rows=rows)


class PhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.packet=phase.preflight(ROOT)

    def test_preflight_is_passive_fixed24_and_hash_bound(self):
        code="import sys;from pathlib import Path;from trottertracks import pr_r_normalization_phase0 as p;x=p.preflight(Path.cwd());assert len(x['upstream_candidates'])==24;assert not any(k in sys.modules for k in ('numpy','scipy','qiskit'))"
        env=os.environ.copy();env['PYTHONPATH']='src:scripts';subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,check=True)
        with patch.object(phase.screen,'screen_normalization',side_effect=AssertionError('No real normalization')):
            packet=phase.preflight(ROOT);plan=phase.public_plan(packet)
        self.assertEqual(len(plan['requests']),24);self.assertEqual(len(plan['input_hashes']),7)
        self.assertTrue(all(r['normalization']['log_B'] is None for r in plan['requests']))
        self.assertNotIn('/private/',json.dumps(plan));self.assertIsNone(plan['shot_inflation_budget'])
        self.assertEqual(sum(len(r['structure']['allocation']) for r in plan['requests']),144)

    def test_fixed_candidate_order_and_H6_allocation_match_upstream(self):
        packet=self.packet
        for system,mapping in phase.MENU.items():
            rows=packet['proposals'][system]['rows']
            self.assertEqual([(r['m'],r['R_outer']) for r in rows],[(m,R) for m,values in mapping.items() for R in values])
            for row in rows:
                self.assertEqual(row['structure']['event_count'],row['q_PF']*row['R_outer'])
                self.assertEqual(sum(row['structure']['allocation']),row['R_outer'])
                self.assertEqual(len(row['structure']['allocation']),1 if system=='H4' else 11)
                if system=='H6' and row['m']==12:self.assertEqual((row['q_RPE'],row['q_PF']),(4096,256))

    def test_source_input_and_scope_mismatches_fail_before_science(self):
        with patch.object(phase,'sha',return_value='FOREIGN'):
            with self.assertRaisesRegex(ValueError,'hash differs'):phase.preflight(ROOT)
        original=phase.MENU
        wrong=deepcopy(original);wrong['H4'][0]=[2,4,16]
        with patch.object(phase,'MENU',wrong):
            with self.assertRaisesRegex(ValueError,'R menu differs'):phase.preflight(ROOT)

    def test_saved_tail_value_is_copied_without_recomputing_norm(self):
        packet=self.packet
        for system,tail in packet['tails'].items():
            spec=packet['input_files'][system+':'+system+'_context.json'];context=json.loads(Path(spec['path']).read_bytes())
            self.assertEqual(tail['lambda_R'],context['prefixes'][str(tail['prefix'])]['lambda_R'])
            self.assertEqual(tail['binding'],context['binding'])
            self.assertEqual(tail['source']['qualification'],'COPIED_SAVED_VALUE_NOT_RECOMPUTED')
        self.assertEqual(packet['historical_status']['H6']['status'],'H6_NOT_AUTHORIZED')

    def test_exp_format_handles_overflow_underflow_and_unavailable_without_N(self):
        self.assertEqual(phase.exp_display(1000.)['status'],'OVERFLOW_BINARY64_KEEP_LOG')
        self.assertIsNone(phase.exp_display(1000.)['value'])
        self.assertEqual(phase.exp_display(-1000.),dict(value=0.,status='UNDERFLOW_TO_ZERO_KEEP_LOG'))
        self.assertEqual(phase.exp_display(None)['status'],'UNAVAILABLE_NORMALIZATION')
        self.assertEqual(phase.exp_display(0.),dict(value=1.,status='FINITE_BINARY64'))

    def test_artificial_24_candidate_normalization_and_persistence(self):
        packet=artificial(self.packet);calls=[]
        def scorer(root,proposal,tail,**kwargs):
            assert tail['lambda_R']==.5 and tail['source']['fixture']=='ARTIFICIAL_TAIL_NOT_MOLECULAR'
            def finite(tau,K):calls.append((tau,K));return SimpleNamespace(exact_finite_distribution=1+tau*tau)
            return phase.screen.screen_normalization(root,proposal,tail,finite_distribution=finite,**kwargs)
        with tempfile.TemporaryDirectory() as t,patch.object(phase.screen.p,'evaluate',side_effect=AssertionError('No G')):
            out=Path(t)/'ARTIFICIAL';result=phase.execute(packet,ROOT,'ARTIFICIAL_ONLY',out,score_provider=scorer)
            self.assertEqual(result['status'],'PHASE0_NORMALIZATION_RECORDED');self.assertEqual(len(calls),144)
            self.assertEqual(len(result['records']),24);self.assertTrue((out/'NORMALIZATION.csv').is_file())
            self.assertTrue(all(r['N'] is None and r['C'] is None and r['G'] is None and r['scientific_candidate_PASS_FAIL'] is None for r in result['records']))
            self.assertTrue(all(r['attenuation']['status']=='FINITE_BINARY64' for r in result['records']))
            with self.assertRaises(FileExistsError):phase.execute(packet,ROOT,'ARTIFICIAL_ONLY',out,score_provider=scorer)
            self.assertEqual(len(calls),144)

    def test_mock_provider_called_only_twice_for_fixed12_each_no_extra_candidates(self):
        packet=artificial(self.packet);scorer=Mock(side_effect=mocked_score)
        with tempfile.TemporaryDirectory() as t:
            result=phase.execute(packet,ROOT,'MOCK_ONLY',Path(t)/'MOCK',score_provider=scorer)
        self.assertEqual(scorer.call_count,2);self.assertEqual([len(c.args[1]['rows']) for c in scorer.call_args_list],[12,12])
        self.assertEqual(result['scope_observed']['normalization_rows'],24)
        self.assertTrue(all(r['phase_radius_N_status']=='NOT_EVALUATED' for r in result['records']))

    def test_expanded_result_or_law_is_rejected_without_retry(self):
        packet=artificial(self.packet)
        for mutate in ('expand','law'):
            def bad(root,proposal,tail,**kw):
                out=mocked_score(root,proposal,tail,**kw)
                if mutate=='expand':out['rows'].append(deepcopy(out['rows'][0]))
                else:out['rows'][0]['request']['law_binding']['K']=0
                return out
            scorer=Mock(side_effect=bad)
            with tempfile.TemporaryDirectory() as t:result=phase.execute(packet,ROOT,'MOCK_ONLY',Path(t)/'FAIL',score_provider=scorer)
            scorer.assert_called_once();self.assertEqual(result['status'],'PHASE0_STOP_TECHNICAL_OR_INPUT')
            self.assertEqual(result['records'],[])

    def test_numerical_unresolved_and_input_loss_remain_non_successful_evidence(self):
        packet=artificial(self.packet)
        def unresolved(root,proposal,tail,**kw):
            out=mocked_score(root,proposal,tail,**kw)
            for r in out['rows']:r['normalization'].update(status='NUMERICAL_NORMALIZATION_UNRESOLVED',log_B=None,optimistic_shot_inflation_log=None)
            return out
        with tempfile.TemporaryDirectory() as t:
            result=phase.execute(packet,ROOT,'MOCK',Path(t)/'UNRESOLVED',score_provider=unresolved)
            self.assertEqual(result['status'],'PHASE0_NORMALIZATION_NUMERICAL_UNRESOLVED')
            self.assertTrue(all(r['attenuation']['value'] is None for r in result['records']))
            self.assertFalse(json.loads((Path(t)/'UNRESOLVED/HANDOFF.json').read_bytes())['normalization_evidence_valid'])
            with patch.object(phase,'sha',side_effect=FileNotFoundError('MOCK_MISSING')):
                result=phase.execute(packet,ROOT,'MOCK',Path(t)/'INPUT_LOSS',score_provider=mocked_score)
            self.assertEqual(result['status'],'PHASE0_STOP_SOURCE_OR_INPUT_CHANGED');self.assertEqual(len(result['records']),24)
            self.assertFalse(json.loads((Path(t)/'INPUT_LOSS/HANDOFF.json').read_bytes())['normalization_evidence_valid'])

    def test_reused_math_and_historical_plan_are_unchanged(self):
        for path in phase.REUSED_FILES:self.assertEqual((ROOT/path).read_bytes(),phase.blob(ROOT,phase.UPSTREAM,path))
        for path in [phase.UP_DIR+'EXECUTION_PLAN.json','docs/research/h4_df10_rpe_batch_results_20261110/N_HANDOFF.json','docs/research/h6_df19_rpe_m8_12_results_20261110/run01/result.json']:
            self.assertEqual(subprocess.check_output(['git','rev-parse','HEAD:'+path],cwd=ROOT),subprocess.check_output(['git','rev-parse',phase.UPSTREAM+':'+path],cwd=ROOT))


if __name__=='__main__':unittest.main()
