"""Six fixed PF Q3 calls consuming A's completed packets; no N search/MC/G."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


H = load_script('_previous_q3_helpers', 'scripts/run_pauli_q3_three_quarter.py')
A = load_script('_completed_pf_locators', 'scripts/accept_completed_pf_shortlist.py')
R, T = H.R, H.T
BASE = '50be68b803a90fd893a1499ebce74e033225144d'
DIRECTORY = 'docs/research/pf_shortlist_q3_fixed_results_20261115'
INPUT = DIRECTORY+'/INPUTS.json'
VECTOR = [59,55,51,47,43,39,35,31,27,23,19,15,11]
LABELS = ['FULL_PF_S2_16h_over_7','FULL_PF_Y4_16h_over_3','FULL_PF_M5_16h',
          'B0I_P7_S2_16h_over_7','B0I_P7_Y4_16h_over_3','B0I_P7_M5_16h']
FAILURE = None


def check_source():
    H.check_source()
    for path in ('scripts/run_pauli_q3_three_quarter.py', 'scripts/accept_completed_pf_shortlist.py'):
        if (ROOT/path).read_bytes() != R.blob(BASE, path):
            raise ValueError('Reused metadata/threshold helper SOURCE differs: '+path)


def metadata(case):
    config = A.selected(case['configuration']); packet = A.selected(case['signal_packet'])
    if config['id'] != case['configuration_id'] or packet['configuration_id'] != config['id']:
        raise ValueError('Original PF configuration/SOURCE mismatch')
    if (T.B.comparison_method(config['spec']) not in ('FULL_PF', 'TRUNCATED_PF')
            or config['spec']['law']['model'] != 'deterministic'
            or config['spec']['finite_K'] is not None or config['spec']['kappa'] is not None):
        raise ValueError('Deterministic PF contract required; no PR/RTE law')
    for row, mean in zip(config['rounds'], packet['means']):
        norm = mean['normalization']
        if (mean['raw'] != mean['corrected'] or norm['B'] != 1 or norm['log_B'] != 0
                or norm['attenuation'] != 1 or row['R_outer'] is not None
                or row['structure']['event_count'] != 0):
            raise ValueError('Saved deterministic raw/normalizer/schedule mismatch')
    target = packet['target']
    if (target['reference_solver_energy_ha'] != -2.1663874486347634
            or target['ground_state_certified'] is not False
            or target['truth_proxy_adopted_for_new_simulation'] is not False):
        raise ValueError('Saved target/qualification differs')
    binding = T.create(ROOT, config, original_source=case['configuration'])
    if binding['total_configuration']['id'] != case['total_view_id']:
        raise ValueError('Fixed total-view ID differs')
    if case.get('binding_id') is not None and binding['id'] != case['binding_id']:
        raise ValueError('Fixed decoder binding differs')
    view = T.bind_signal(ROOT, binding, packet, original_packet_source=case['signal_packet'])
    return binding, view


def prepare():
    check_source()
    acceptance = A.build()  # Metadata only; reuses the prior A intake and C alignment.
    index = A.selected(acceptance['A_index'])
    if [v['candidate'] for v in index['packets']] != LABELS:
        raise ValueError('Exactly the six fixed shortlist configurations required')
    old = json.loads(R.blob(BASE, 'docs/research/pauli_q3_three_quarter_20261115/INPUTS.json'))
    cases = []
    for item in index['packets']:
        case = deepcopy(item); case['label'] = case.pop('candidate')
        binding, view = metadata(case); target = view['signal_packet']['target']
        case.update(binding_id=binding['id'], allocation=dict(
            allocation_id=case['label']+'_DET_PAPER_SHAPED_CONTROL_01', shot_policy='USER_SUPPLIED',
            shots=[dict(m=m, N_X=n, N_Y=n) for m,n in enumerate(VECTOR)]),
            evaluation_target=dict(kind='SAVED_FULL_DF_H_SOLVER_CONDITIONAL_PROXY',
                energy_ha=-2.1663874486347634, source=deepcopy(target['source']),
                truth_proxy_source=deepcopy(target['truth_proxy_source']),
                original_target_source=deepcopy(target['source']), binding=deepcopy(target['binding']),
                qualification_scope='RAW_MODEL_DIAGNOSTIC_ONLY_NOT_CERTIFIED_GROUND_ENERGY'))
        cases.append(case)
    contract = deepcopy(old['statistical_contract'])
    contract['id'] = 'PF_SHORTLIST_SIX_TOTAL_Q3_FIXED_CONTROLS'
    contract['simulation_policy'].update(target_law='deterministic',
        interpretation='IID_BINOMIAL_MEASUREMENTS_OF_SAVED_DETERMINISTIC_PF_RAW_NO_RTE_OR_MC')
    options = deepcopy(old['options'])
    options['model_loss_contract']['scope'] = 'SIX_PF_TOTAL_DECODER_ALL_EVENT_SQUARED_ERROR_CONTROLS_ONLY'
    adopted = deepcopy(old['adopted_evaluation'])
    adopted['MSE_upper'] = T.B.a.exact_value(H.Fraction(adopted['epsilon_E_decimal'])**2)
    adopted['measurement_law'] = 'SAVED_DETERMINISTIC_PF_RAW_IID_EXACT_AFFINE'
    bounds = H.finite_bounds(VECTOR)
    if (bounds['maximum_round_outcomes'] > options['maximum_round_outcomes']
            or bounds['verification_pairs_upper'] > options['maximum_verification_pairs']
            or bounds['transition_pairs_upper'] > options['total_state_refinement']['maximum_transition_pairs']):
        raise ValueError('Finite count-only scope exceeded; do not start')
    inputs = dict(B_source=BASE, kernel_source=old['kernel_source'], C_source=T.C.SOURCE,
        A_index=acceptance['A_index'], A_intake=acceptance['reused_B_intake'], C_alignment=acceptance['C_alignment'],
        cases=cases, statistical_contract=contract, options=options, adopted_evaluation=adopted,
        finite_scope_preflight=bounds, acceleration=deepcopy(old['acceleration']),
        maximum_scientific_calls=6, concurrent_configurations=6,
        scope='SIX_FIXED_PF_Q3_DIAGNOSTICS_ONLY_NO_PAULI_OR_D_PRICE_RUNS',
        sufficient_N=None, global_RMSE_bound=None, G=None, G_qualified=None, winner=None)
    (ROOT/DIRECTORY).mkdir(exist_ok=True)
    R.write_new(ROOT/INPUT, inputs)
    print('Six PF metadata bindings fixed; no risk called. Count bounds:', bounds, flush=True)


def init_worker(flag):
    global FAILURE
    FAILURE = flag


def monitor_pending(output, pending, start, peaks):
    # pending maps Future -> label; resource paths must use its values.
    for label in pending.values():
        R.resources(output/label, ['job'], start, peaks[label])


def worker(case, inputs, source, output):
    output = Path(output); start = time.perf_counter(); cpu = time.process_time()
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN); calls = 0
    R.write_new(output/'START_job.json', dict(label=case['label'], pid=os.getpid(), source=source, started_UTC=R.utc()))
    def event(name, **values):
        with (output/'STAGES.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(stage=name, UTC=R.utc(), wall_seconds=time.perf_counter()-start, **values))+'\n')
    result = dict(label=case['label'], source=source, configuration_source=case['configuration'],
        signal_packet_source=case['signal_packet'], allocation=deepcopy(case['allocation']),
        MC_calls=0, old_partial_DP_calls=0, price_calls=0, additional_allocations=False, deltaE_added=False,
        ground_certified=False, sampler_law_transfer_certified=False, numerical_certificate=None,
        sufficient_N=None, global_RMSE_bound=None, G=None, G_qualified=None, winner=None, mandatory_stop=True)
    engine = R.Observed(output, 'job', workers=inputs['acceleration']['workers_per_configuration'],
                        outcome_batch=inputs['acceleration']['outcome_batch'])
    try:
        check_source(); binding, view = metadata(case)
        allocation = deepcopy(case['allocation']); allocation['source'] = deepcopy(source)
        contract = deepcopy(inputs['statistical_contract']); contract['source'] = deepcopy(source)
        contract['simulation_policy']['source'] = deepcopy(source)
        contract['summary_options']['unresolved_loss']['source'] = deepcopy(source)
        options = deepcopy(inputs['options']); options['source'] = deepcopy(source)
        options['model_loss_contract']['source'] = deepcopy(source)
        options['total_state_refinement']['source'] = deepcopy(source)
        original_total, original_lemma = T.Q._total, T.Q.lemma_bound
        def total_gate(entry, partial, opts):
            R.write_new(output/'PARTIAL_COARSE.json', partial)
            if (partial.get('decoder_certificate') or {}).get('status') != T.Q.VERIFIED:
                raise ValueError('RHO_CLOSURE_UNKNOWN:'+str(partial.get('reason')))
            event('GOOD_CLOSURE_VERIFIED')
            return original_total(entry, partial, opts)
        def lemma_gate(*args, **kwargs):
            value = original_lemma(*args, **kwargs); R.write_new(output/'PARTIAL_LEMMA.json', value)
            if value.get('MSE_upper') is None: raise ValueError('LEMMA_UNKNOWN:'+str(value.get('reason')))
            event('LEMMA_SAVED'); return value
        def dp_gate(*args, **kwargs):
            event('TOTAL_DP_START'); value = engine(*args, **kwargs)
            R.write_new(output/'PARTIAL_TOTAL_DP.json', value)
            if value.get('MSE_upper') is None: raise ValueError('TOTAL_DP_UNKNOWN:'+str(value.get('reason')))
            if value['transition_pairs'] > options['total_state_refinement']['maximum_transition_pairs'] or any(
                    r['retained_states'] > options['total_state_refinement']['maximum_frontier_states'] for r in value['rounds'][:-1]):
                raise ValueError('TOTAL_DP_FINITE_SCOPE_EXCEEDED')
            event('TOTAL_DP_RETURNED', transition_pairs=value['transition_pairs']); return value
        with ExitStack() as stack:
            for module, name in ((T.Q.S,'simulate'), (T.Q.S,'simulate_prepared'), (T.K,'diagnose'),
                                 (T.p,'run_missing'), (T.B,'cost'), (T.B,'select')):
                stack.enter_context(patch.object(module, name, side_effect=AssertionError('Outside scope: '+name)))
            for name, hook in (('_total',total_gate), ('lemma_bound',lemma_gate), ('total_state_dp',dp_gate)):
                stack.enter_context(patch.object(T.Q, name, hook))
            if FAILURE is not None and FAILURE.is_set(): raise ValueError('Sibling failure; science call not started')
            event('SCIENTIFIC_CALL_START'); calls += 1
            records = T.evaluate(ROOT, binding, view['signal_packet'], [allocation],
                evaluation_target=case['evaluation_target'], statistical_contract=contract, source=source, options=options)
        if len(records) != 1: raise ValueError('One record required')
        record = records[0]; risk = record['diagnostic']['deterministic_risk']
        if risk.get('MSE_upper') is None: raise ValueError('TOTAL_RISK_UNKNOWN:'+str(risk.get('reason')))
        if any(record.get(k) is not None for k in ('n_packet','sufficient_N','global_RMSE_bound','G','G_qualified')):
            raise ValueError('Forbidden N/G qualification')
        assessment = H.assess(risk, inputs['adopted_evaluation'], source)
        result.update(status=assessment['status'], record=record, adopted_q3_assessment=assessment)
        if assessment['status'] == 'UNKNOWN': raise ValueError('Missing model bound')
        event('EVALUATION_COMPLETE', status=assessment['status'])
    except Exception as exc:
        if FAILURE is not None: FAILURE.set()
        result.update(status='UNKNOWN', reason=str(exc), error_type=type(exc).__name__, automatic_retry=False)
        event('UNKNOWN_STOP', reason=str(exc))
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = children.ru_utime+children.ru_stime-child_start.ru_utime-child_start.ru_stime
    result.update(scientific_evaluation_calls=calls, performance=dict(wall_seconds=time.perf_counter()-start,
        cpu_seconds=time.process_time()-cpu, child_cpu_seconds=child_cpu,
        process_tree_cpu_seconds=time.process_time()-cpu+child_cpu,
        peak_worker_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        max_single_child_RSS_KiB=children.ru_maxrss, pid=os.getpid()),
        acceleration=dict(backend=deepcopy(inputs['acceleration']), rounds=deepcopy(engine.observations)))
    return result


def execute():
    check_source(); sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    for path in (INPUT, 'scripts/'+Path(__file__).name):
        if (ROOT/path).read_bytes() != R.blob(sha, path): raise ValueError('Unfixed runner/INPUTS')
    inputs = json.loads((ROOT/INPUT).read_text())
    if [r['label'] for r in inputs['cases']] != LABELS: raise ValueError('Six fixed cases required')
    output = ROOT/DIRECTORY/'run01'; output.mkdir()
    for label in LABELS: (output/label).mkdir()
    source = dict(commit=sha, path=INPUT, caller='scripts/'+Path(__file__).name)
    R.write_new(output/'EXECUTION.json', dict(source=source, started_UTC=R.utc(), maximum_scientific_calls=6))
    start = time.perf_counter(); context = mp.get_context('fork'); failure = context.Event()
    completed=[]; unknown=[]; errors=[]; calls=0; peaks={label:{} for label in LABELS}
    with ProcessPoolExecutor(max_workers=inputs['concurrent_configurations'], mp_context=context,
                             initializer=init_worker, initargs=(failure,)) as pool:
        pending={pool.submit(worker, case, inputs, source, str(output/case['label'])):case['label'] for case in inputs['cases']}
        while pending:
            done,_=wait(pending, timeout=2, return_when=FIRST_COMPLETED)
            for future in done:
                label=pending.pop(future)
                try:
                    result=future.result(); result['performance']['sampled_peak_process_tree_RSS_KiB']=peaks[label].get('job')
                    R.write_new(output/label/'RESULT.json',result); completed.append(label)
                    calls+=result['scientific_evaluation_calls']
                    if result['status']=='UNKNOWN': unknown.append(label); failure.set()
                    print(label+': '+result['status'],flush=True)
                except Exception as exc:
                    failure.set(); errors.append(dict(label=label,type=type(exc).__name__,message=str(exc)))
            monitor_pending(output, pending, start, peaks)
    R.write_new(output/'COMPLETION.json',dict(status='SIX_RESULTS_SAVED' if not unknown and not errors else 'UNKNOWN_STOP_NO_RETRY',
        completed=sorted(completed),unknown=unknown,errors=errors,scientific_evaluation_calls_in_saved_results=calls,
        wall_seconds=time.perf_counter()-start,finished_UTC=R.utc(),
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute'))
    (prepare if parser.parse_args().mode=='prepare' else execute)()
