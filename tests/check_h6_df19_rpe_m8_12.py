"""Fixed-menu wrapper and artificial endpoint tests. No molecular arrays."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import run_h6_df19_rpe_m8_12 as batch
from trottertracks import pf_pr_rpe_pipeline as p


def mock_records(packet):
    from check_pf_pr_rpe_pipeline import fake_shot
    records=[]
    for row in packet['rows']:
        n=fake_shot(packet['task'],row);n['diagnostics']=dict(synthetic_only=True)
        records.append(n)
    return records


class RemainingTests(unittest.TestCase):
    def test_fixed10_menu_rejects_scope_policy_binding_and_recipe_changes(self):
        ready=batch.prepare();batch.validate_menu(ready['plan'],ready['task'])
        self.assertEqual(tuple(x['candidate']['id'] for x in ready['rows']),batch.IDS)
        self.assertEqual([r['q_PF'] for r in ready['rows']],[16,16,32,32,64,64,128,128,256,256])
        self.assertEqual([r['q_RPE'] for r in ready['rows']],[256,256,512,512,1024,1024,2048,2048,4096,4096])
        for change in [lambda x:x['rows'].pop(),
                lambda x:x['rows'][0]['candidate'].update(q=256),
                lambda x:x['configuration_reuse'].update(alpha_axis=.025),
                lambda x:x.update(norm_and_normalization_gate=1e-8),
                lambda x:x['planned_scientific_scope'].update(finite_mean_candidates=20),
                lambda x:x['PR_allocation'].__setitem__(0,2),
                lambda x:x['scientific_binding'].update(state='FOREIGN')]:
            bad=deepcopy(ready['plan']);change(bad)
            with self.assertRaises(ValueError):batch.prepare(plan=bad)

    def test_prepare_is_N_only_and_preserves_original16(self):
        with patch.object(p,'saved_direct_prices',side_effect=AssertionError('No C')), \
             patch.object(p,'saved_model_prices',side_effect=AssertionError('No model')), \
             patch.object(p,'evaluate',side_effect=AssertionError('No G')):
            ready=batch.prepare()
        self.assertEqual(len(ready['saved_N']),16)
        self.assertEqual(sorted({r['identity']['m'] for r in ready['saved_N']}),list(range(8)))
        self.assertTrue(all('compiled_prices' not in n['diagnostics'] for n in ready['saved_N']))
        self.assertEqual(len(ready['previous_source_files']),16)

    def test_coverage_distinguishes_PASS_FAIL_missing_and_invalid_provenance(self):
        ready=batch.prepare();records=mock_records(ready)
        out=batch.coverage(ready,records,'SYNTHETIC_ONLY')
        self.assertEqual(out['rows'][:16],ready['previous']['rows'])
        self.assertEqual(len(out['rows']),26);self.assertTrue(out['all_13_N_rounds_conditional_pass'])
        self.assertFalse(out['full_RPE_qualified']);self.assertIsNone(out['total_resources'])
        broken=deepcopy(records);broken[0]['plan']['N_X']=None
        incomplete=batch.coverage(ready,broken,'SYNTHETIC_ONLY')
        self.assertFalse(incomplete['all_13_N_rounds_conditional_pass'])
        self.assertEqual(incomplete['rows'][16]['status'],'UNQUALIFIED_N_MISSING')
        records[0]['plan'].update(status='UNQUALIFIED_ROUND',N_X=None,N_Y=None)
        out=batch.coverage(ready,records[:-1],'SYNTHETIC_ONLY')
        self.assertFalse(out['all_13_N_rounds_conditional_pass'])
        self.assertIsNone(out['rows'][16]['N_X']);self.assertEqual(out['rows'][-1]['status'],'NOT_EVALUATED_AFTER_STOP')
        out=batch.coverage(ready,records,'SYNTHETIC_ONLY',valid=False)
        self.assertTrue(all(x['N_X'] is None for x in out['rows'][16:]))
        self.assertTrue(all(x['C'] is None and x['G'] is None for x in out['new_price_handoff']))

    def test_one_mock_batch_persists10_records_without_C_or_old_N_recalculation(self):
        ready=batch.prepare();provider=Mock(return_value=mock_records(ready))
        provider.last_run=dict(synthetic_only=True,reference_state_actions=0)
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(p,'evaluate',side_effect=AssertionError('No G')):
            out=Path(temp)/'MOCK_ONLY'
            result=batch.execute(ready,provider,'SYNTHETIC_ONLY',out)
            self.assertEqual(result['status'],'M8_12_N_BATCH_EVALUATED')
            self.assertEqual(result['saved_round_re_evaluations'],0)
            self.assertEqual(result['price_calls'],0);self.assertEqual(result['G_evaluations'],0)
            self.assertEqual(len(list(out.glob('*.json'))),13)
            provider.assert_called_once();self.assertEqual(len(provider.call_args.args[1]),10)
            self.assertEqual(json.loads((out/'result.json').read_bytes())['previous_rounds'],ready['previous']['rows'])
            with self.assertRaises(FileExistsError):batch.execute(ready,provider,'SYNTHETIC_ONLY',out)
            provider.assert_called_once()

    def test_numerical_FAIL_and_reference_exception_are_saved_without_retry(self):
        ready=batch.prepare();records=mock_records(ready)
        records[0]['plan'].update(status='NUMERICAL_GATE_FAILURE',N_X=None,N_Y=None)
        records[0]['diagnostics']['reason']='MOCK_ONLY_GATE'
        provider=Mock(return_value=records);provider.last_run=dict(synthetic_only=True,status='STOP_UNQUALIFIED_N')
        with tempfile.TemporaryDirectory() as temp:
            result=batch.execute(ready,provider,'SYNTHETIC_ONLY',Path(temp)/'MOCK_FAIL')
            self.assertEqual(result['status'],'M8_12_STOP_UNQUALIFIED_N');provider.assert_called_once()
            self.assertIsNone(result['new_records'][0]['plan']['N_X'])
            reference_fail=Mock(side_effect=ValueError('MOCK_REFERENCE_GATE'))
            reference_fail.last_run=dict(status='STOP_REFERENCE_GATE',reference_times=[25.6],references={},synthetic_only=True)
            out=Path(temp)/'MOCK_REFERENCE_FAIL'
            result=batch.execute(ready,reference_fail,'SYNTHETIC_ONLY',out)
            self.assertEqual(result['status'],'M8_12_STOP_TECHNICAL_OR_REFERENCE');reference_fail.assert_called_once()
            coverage=json.loads((out/'COVERAGE.json').read_bytes())
            self.assertFalse(coverage['all_13_N_rounds_conditional_pass'])
            self.assertTrue(all(r['N_X'] is None for r in coverage['rows'][16:]))

    def test_artificial_q16_q256_reuse_independent_finite_mean(self):
        from check_h6_df19_rpe_d2 import toy,candidate,ready,old
        from trottertracks import h6_df19_rpe_d2 as backend
        for q in (16,256):
            ctx=toy();c=candidate(T=.175*q,q=q);c['R_outer']=16
            reference=ready(ctx,c);actual=backend.evolve(ctx,c,reference)
            with patch.object(old,'weights',backend.schedule.weights):
                expected=old.compare_prefix(ctx.state,[old.spectrum(a) for a in ctx.matrices[:2]],
                    old.spectrum(ctx.tails[1]['matrix']),.21,.5,c['formula'],q,T=.175*q,
                    total_steps=16,cutoffs=(2,))['finite_rows'][0]
            for key,target in [('corrected',expected['corrected_signal']),('raw',expected['raw_signal'])]:
                value=actual['signal'][key]
                self.assertLess(abs(complex(value['real'],value['imag'])-target),2e-11)
            self.assertAlmostEqual(actual['signal']['log_B'],expected['log_B'],places=12)
            self.assertEqual(actual['allocation'],[1,2,1,2,1,2,1,2,1,2,1])
            self.assertEqual(actual['event_count'],16*q)
            self.assertLess(actual['signal']['allowance_evidence']['raw_corrected_vector_closure'],1e-9)

    def test_old_m7_guard_not_expanded(self):
        import run_h6_df19_rpe_m7 as m7
        ready=batch.prepare();old=json.loads((ROOT/m7.PLAN_PATH).read_bytes())
        m7.validate_m7(old)
        changed=deepcopy(old);changed['rows']=ready['rows']
        with self.assertRaises(ValueError):m7.validate_m7(changed)


if __name__=='__main__':unittest.main()
