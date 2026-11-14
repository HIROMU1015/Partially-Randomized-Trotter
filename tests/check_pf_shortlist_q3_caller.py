"""Artificial caller risks and saved metadata only; no real H4 Q3 calls."""
from copy import deepcopy
from concurrent.futures import Future
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from check_rpe_total_binding import pauli, T, MOCK
spec=importlib.util.spec_from_file_location('_pf_q3_caller',ROOT/'scripts/run_pf_shortlist_total_q3_fixed.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def artificial():
    args,_=pauli(); bridge=T.create(ROOT,args['config'],original_source=MOCK)
    view=T.bind_signal(ROOT,bridge,args['signal_packet'])
    case=dict(label='ARTIFICIAL_ONLY',configuration=MOCK,signal_packet=MOCK,
        allocation=args['allocations'][0],evaluation_target=args['evaluation_target'])
    inputs=dict(statistical_contract=args['statistical_contract'],options=args['options'],
        acceleration=dict(workers_per_configuration=2,outcome_batch=1,source=MOCK),
        adopted_evaluation=dict(MSE_upper=999,epsilon_E_decimal='ARTIFICIAL_ONLY'))
    return case,inputs,bridge,view


class CallerTests(unittest.TestCase):
    def test_monitor_uses_candidate_values_not_future_keys(self):
        future=Future();pending={future:'ARTIFICIAL_ONLY'};peaks={'ARTIFICIAL_ONLY':{}}
        with patch.object(M.R,'resources')as observe:
            M.monitor_pending(Path('/tmp/artificial_pf_monitor'),pending,0.,peaks)
        observe.assert_called_once_with(Path('/tmp/artificial_pf_monitor/ARTIFICIAL_ONLY'),
                                        ['job'],0.,peaks['ARTIFICIAL_ONLY'])

    def test_fixed_integer_vector_and_deterministic_contract(self):
        inputs=json.loads((ROOT/M.INPUT).read_text())
        self.assertEqual([c['label'] for c in inputs['cases']],M.LABELS)
        self.assertEqual(inputs['statistical_contract']['simulation_policy']['target_law'],'deterministic')
        for case in inputs['cases']:
            self.assertEqual([r['N_X']for r in case['allocation']['shots']],M.VECTOR)
            self.assertEqual([r['N_Y']for r in case['allocation']['shots']],M.VECTOR)
        self.assertEqual(M.H.finite_bounds(M.VECTOR),inputs['finite_scope_preflight'])

    def test_artificial_worker_reuses_decoder_and_threshold_without_N_or_G(self):
        case,inputs,bridge,view=artificial();old=deepcopy((case,inputs,view))
        with tempfile.TemporaryDirectory()as out,patch.object(M,'metadata',return_value=(bridge,view)):
            result=M.worker(case,inputs,MOCK,out)
        self.assertEqual(result['status'],'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD')
        self.assertEqual(result['scientific_evaluation_calls'],1)
        for key in ('sufficient_N','global_RMSE_bound','G','G_qualified'):
            self.assertIsNone(result[key])
        self.assertEqual(result['MC_calls'],0);self.assertEqual(result['price_calls'],0)
        self.assertEqual((case,inputs,view),old)

    def test_closure_failure_saved_without_retry(self):
        case,inputs,bridge,view=artificial()
        def fail(*args,**kwargs):
            return T.Q._total(None,dict(decoder_certificate=dict(status='UNKNOWN')),None)
        with tempfile.TemporaryDirectory()as out,patch.object(M,'metadata',return_value=(bridge,view)),\
                patch.object(T,'evaluate',side_effect=fail)as call:
            result=M.worker(case,inputs,MOCK,out)
            self.assertTrue((Path(out)/'PARTIAL_COARSE.json').exists())
        self.assertEqual(result['status'],'UNKNOWN');self.assertEqual(call.call_count,1)
        self.assertIn('RHO_CLOSURE_UNKNOWN',result['reason'])

    def test_failure_flag_prevents_not_started_scientific_call(self):
        case,inputs,bridge,view=artificial()
        flag=M.mp.get_context('fork').Event();flag.set()
        with tempfile.TemporaryDirectory()as out,patch.object(M,'metadata',return_value=(bridge,view)),\
                patch.object(M,'FAILURE',flag),patch.object(T,'evaluate',side_effect=AssertionError('Must not call'))as call:
            result=M.worker(case,inputs,MOCK,out)
        self.assertEqual(result['status'],'UNKNOWN');self.assertEqual(result['scientific_evaluation_calls'],0)
        self.assertEqual(call.call_count,0)

    def test_swapped_saved_configuration_rejected_before_risk(self):
        cases=json.loads((ROOT/M.INPUT).read_text())['cases']
        bad=deepcopy(cases[0]);bad['signal_packet']=deepcopy(cases[-1]['signal_packet'])
        with patch.object(T,'evaluate',side_effect=AssertionError('No real risk'))as call:
            with self.assertRaisesRegex(ValueError,'configuration/SOURCE'):M.metadata(bad)
        self.assertEqual(call.call_count,0)


if __name__=='__main__':unittest.main(verbosity=2)
