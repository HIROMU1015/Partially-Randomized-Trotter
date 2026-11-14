"""Report already saved kernel outputs after driver failure. NEVER runs risk."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import importlib.util
import json
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_pf_fixed_metadata',ROOT/'scripts/run_pf_shortlist_total_q3_fixed.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
DIRECTORY=ROOT/M.DIRECTORY


def read(path):return json.loads(path.read_text())


def recover():
    inputs=read(DIRECTORY/'INPUTS.json');run=DIRECTORY/'run01';execution=read(run/'EXECUTION.json')
    failure=read(run/'DRIVER_ERROR.json');completed=[];calls=0
    # Only metadata binding, exact comparisons, and the EXISTING Q3 filter.
    with ExitStack()as stack:
        for module,name in ((M.T,'evaluate'),(M.T.Q,'evaluate'),(M.T.Q.D,'evaluate'),
                            (M.T.Q.D,'round_distribution'),(M.T.Q.D,'binomial_axis'),
                            (M.T.Q,'total_state_dp'),(M.T.Q.S,'simulate'),(M.T.Q.S,'simulate_prepared'),
                            (M.T.B,'cost'),(M.T.B,'select')):
            stack.enter_context(patch.object(module,name,side_effect=AssertionError('Recovery cannot compute: '+name)))
        M.check_source()
        for case in inputs['cases']:
            label=case['label'];folder=run/label
            if (folder/'RESULT.json').exists():raise ValueError('Do not replace an original result')
            stages=[json.loads(x)for x in(folder/'STAGES.jsonl').read_text().splitlines()]
            count=sum(r['stage']=='SCIENTIFIC_CALL_START'for r in stages)
            if count!=1 or stages[-1]['stage']!='EVALUATION_COMPLETE':
                raise ValueError('Not a completed single-call scientific worker: '+label)
            coarse=read(folder/'PARTIAL_COARSE.json');lemma=read(folder/'PARTIAL_LEMMA.json')
            dp=read(folder/'PARTIAL_TOTAL_DP.json')
            exact=M.T.Q.a.exact_scalar
            # No new bound/model: DP endpoints were already emitted by the kernel.
            # Check they also attained the min used by the completed worker.
            if not (exact(dp['MSE_upper'])<=exact(lemma['MSE_upper'])
                    and exact(dp['MSE_upper'])<=exact(coarse['MSE_upper'])):
                raise ValueError('Cannot report the original best bound from DP alone')
            risk=dict(status='SAVED_TOTAL_DP_ENDPOINTS_NOT_ORIGINAL_API_RETURN',
                MSE_lower=deepcopy(dp['MSE_lower']),MSE_upper=deepcopy(dp['MSE_upper']),
                MSE_upper_method='SAVED_TOTAL_DP_IS_MIN_OF_ALL_SAVED_VALID_UPPERS',
                moments=deepcopy(dp['moments']),bounds=dict(total_state_dp=dp,lemma=lemma,
                    upper_candidates=dict(partial_coarse=coarse['MSE_upper'],lemma=lemma['MSE_upper'],
                                          total_state_dp=dp['MSE_upper'])))
            assessment=M.H.assess(risk,inputs['adopted_evaluation'],execution['source'])
            if assessment['status']!=stages[-1]['status']:
                raise ValueError('Saved worker classification does not match saved bound')
            binding,view=M.metadata(case)  # SOURCE/identity only, no count/risk calculation.
            value=dict(label=label,status=assessment['status'],source=execution['source'],
                evidence_status='SAVED_KERNEL_COMPONENTS_RECOVERED_ORIGINAL_RETURN_NOT_SAVED',
                recovery_scientific_calls=0,original_api_return_available=False,
                configuration_source=case['configuration'],signal_packet_source=case['signal_packet'],
                allocation=case['allocation'],scientific_evaluation_calls=count,
                model_bound=risk,adopted_q3_assessment=assessment,
                canonical_binding_metadata=binding,
                physical_ids=[r['physical_candidate']['id']for r in view['signal_packet']['means']],
                performance=dict(wall_to_evaluation_complete_seconds=stages[-1]['wall_seconds'],
                    cpu_seconds=None,process_tree_cpu_seconds=None,peak_worker_RSS_KiB=None,
                    sampled_peak_process_tree_RSS_KiB=None,
                    missing_reason='Worker return and driver resource samples were not saved'),
                original_evidence_files=['PARTIAL_COARSE.json','PARTIAL_LEMMA.json','PARTIAL_TOTAL_DP.json','STAGES.jsonl'],
                MC_calls=0,old_partial_DP_calls=0,price_calls=0,deltaE_added=False,
                sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)
            M.R.write_new(folder/'RECOVERED_RESULT.json',value);completed.append(label);calls+=count
    M.R.write_new(run/'COMPLETION.json',dict(status='SCIENTIFIC_COMPONENTS_COMPLETE_DRIVER_SAVE_FAILED',
        operational_status='UNKNOWN_DRIVER_MONITOR_AND_FINAL_RETURN_PERSISTENCE',
        completed_scientific_workers=completed,scientific_evaluation_calls_in_saved_stages=calls,
        recovery_scientific_calls=0,automatic_retry=False,driver_error=failure,
        CPU_and_peak_RSS=None,sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,
        mandatory_stop=True))
    print('Six saved scientific component sets recovered; ZERO new scientific calls. Driver failure retained.')


if __name__=='__main__':recover()
