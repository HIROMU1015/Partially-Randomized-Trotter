"""One LD3 VP total/Q3 call; source-bound Pauli intake, no prices or search."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]
spec = importlib.util.spec_from_file_location('_ld3_saved_worker', ROOT/'scripts/run_pr_df_fullh_total_q3_fixed.py')
M = importlib.util.module_from_spec(spec); sys.modules[spec.name] = M; spec.loader.exec_module(M)
spec = importlib.util.spec_from_file_location('_ld3_saved_observer', ROOT/'scripts/run_full_h_kappa2_scale3o2_q3_fixed.py')
O = importlib.util.module_from_spec(spec); sys.modules[spec.name] = O; spec.loader.exec_module(O)
from trottertracks import rpe_pauli_signal_intake as A
T, R, H = M.T, M.R, M.H
BASE = 'cbc471c913072de45c25a0ed22d51dff8a9b7034'
PAULI = '50be68b803a90fd893a1499ebce74e033225144d'
PAULI_DIR = 'docs/research/pauli_q3_three_quarter_20261115'
DIRECTORY = 'docs/research/pauli_ld3_vp_q3_results_20261116'
INPUT = DIRECTORY+'/INPUTS.json'
CALLER = 'scripts/run_pauli_ld3_vp_q3_fixed.py'
LABEL = 'PAULI_LD3_VP'
CONFIGURATION_ID = 'dd5b881d308f6fc48d3c91d08cec22c5fbbac296f60242e9bfb433c3e0321847'
VECTOR = [48,45,42,39,36,33,29,26,23,19,16,13,9]
ADAPTED = ('rpe_pauli_signal_intake', 'rpe_total_binding', 'rpe_c_interfaces')
SAVED_WORKER = M.worker


def document(commit, path):
    return json.loads(R.blob(commit, path))


def check_source():
    # The three passive adapters are explicitly changed for LD3 and fixed in
    # the execution commit. All numerical kernels/actual workers stay exact.
    names = ('rpe_total_risk', 'rpe_total_decoder', 'rpe_deterministic_risk',
             'rpe_decoder_fast', 'rpe_decoder_statistics', 'rpe_signal_statistics_caller',
             'pr_rpe_rmse_controller', 'rpe_shot_allocation', 'rpe_price_implementation_guard',
             'pf_pr_rpe_pipeline', 'rpe_total_dp_parallel')
    for path in ['src/trottertracks/'+n+'.py' for n in names] + [
            'scripts/run_pf_shortlist_total_q3_fixed.py', 'scripts/run_pr_df_fullh_total_q3_fixed.py',
            'scripts/run_full_h_kappa2_scale3o2_q3_fixed.py', 'scripts/run_pauli_q3_three_quarter.py',
            'scripts/run_pauli_total_q3_accelerated.py']:
        if (ROOT/path).read_bytes() != R.blob(BASE, path):
            raise ValueError('Reused SOURCE differs: '+path)
    for name in ('rpe_total_risk', 'rpe_total_decoder'):
        path = 'src/trottertracks/'+name+'.py'
        if (ROOT/path).read_bytes() != R.blob('710aa69b663ef3313439762d25c80a652a03948f', path):
            raise ValueError('Fixed kernel differs')


def metadata(case):
    api = T.C.interfaces(str(ROOT)).pipeline
    accepted = A.accept_saved(ROOT, integration=api, handoff_source=A.LD3_HANDOFF)['results'][0]
    original = T._selected(ROOT, case['configuration'])
    if original != accepted['configuration'] or original['id'] != CONFIGURATION_ID:
        raise ValueError('LD3 original configuration differs')
    binding = T.create(ROOT, original, original_source=case['configuration'])
    if case.get('binding_id') is not None and (case['binding_id'] != binding['id'] or
            case['total_configuration_id'] != binding['total_configuration']['id']):
        raise ValueError('LD3 total canonical bridge differs')
    view = T.bind_signal(ROOT, binding, accepted['signal_packet'])
    return binding, view


def build_inputs():
    check_source()
    handoff = document(A.LD3_HANDOFF['commit'], A.LD3_HANDOFF['path'])
    old = document(PAULI, PAULI_DIR+'/INPUTS.json')
    case = dict(label=LABEL, configuration=deepcopy(handoff['configuration_source']),
        signal_packet=deepcopy(A.LD3_HANDOFF), original_configuration_id=CONFIGURATION_ID,
        allocation=dict(allocation_id='PAULI_LD3_USER_VP_MATCH_LD2_01', shot_policy='USER_SUPPLIED',
            shots=[dict(m=m, N_X=n, N_Y=n) for m,n in enumerate(VECTOR)]),
        evaluation_target=deepcopy(old['cases'][1]['evaluation_target']))
    binding, view = metadata(case)
    if case['allocation']['shots'] != old['cases'][1]['allocation']['shots']:
        raise ValueError('LD2 VP comparison vector differs')
    case.update(binding_id=binding['id'], total_configuration_id=binding['total_configuration']['id'])
    limits = H.finite_bounds(VECTOR)
    options = old['options']
    if (limits['maximum_round_outcomes'] > options['maximum_round_outcomes'] or
            limits['verification_pairs_upper'] > options['maximum_verification_pairs'] or
            limits['transition_pairs_upper'] > options['total_state_refinement']['maximum_transition_pairs']):
        raise ValueError('Finite scope cannot include this vector')
    return dict(B_source=BASE, cases=[case], maximum_scientific_calls=1, concurrent_configurations=1,
        kernel_source=old['kernel_source'], C_source=T.C.SOURCE,
        statistical_contract=deepcopy(old['statistical_contract']), options=deepcopy(options),
        acceleration=deepcopy(old['acceleration']), adopted_evaluation=deepcopy(old['adopted_evaluation']),
        finite_scope_preflight=limits,
        original_Pauli_Q3_route=dict(commit='f54b41a2e311e44124c9e180e26b01fe11d4eabd',
            path='scripts/run_pauli_q3_three_quarter.py'),
        dictionary_helper=dict(commit=A.LD3_HELPER_SOURCE,
            path='src/trottertracks/df_head_pauli_tail_ld3_signals.py', function='b_intake_dictionary_identity'),
        comparison=dict(commit=PAULI, path=PAULI_DIR+'/run01/RESULT_LD2.json'),
        scope='ONE_LD3_TOTAL_VIEW_AND_ONE_VP_Q3_ONLY_NO_LD2_REEVALUATION_NO_PRICE',
        duplicate_check='SCOPE_AND_PREFLIGHT.md', signal_recomputed=False,
        sufficient_N=None, global_RMSE_bound=None, G=None, G_qualified=None, winner=None)


def prepare():
    inputs = build_inputs(); path = ROOT/DIRECTORY; path.mkdir(exist_ok=True)
    R.write_new(path/'INPUTS.json', inputs)
    binding, view = metadata(inputs['cases'][0])
    R.write_new(path/'TOTAL_BINDING.json', binding)
    R.write_new(path/'TOTAL_SIGNAL_VIEW.json', view['signal_packet'])
    R.write_new(path/'C_SIGNAL_INPUT.json', view['signal_input'])
    print('LD3 passive view fixed:', binding['total_configuration']['id'], inputs['finite_scope_preflight'], flush=True)


def worker(case, inputs, source, output):
    with patch.object(M, 'metadata', metadata), patch.object(M, 'check_source', check_source):
        return SAVED_WORKER(case, inputs, source, output)


@contextmanager
def installed(output):
    # Observation's module was independently loaded. Bind its patch targets to
    # this actual entry, rather than another instance of the saved caller.
    # Corrected after run01; the scientific result remains at SOURCE 7733089b8.
    with patch.object(M, 'worker', worker), patch.object(O, 'M', M), patch.object(O, 'R', R), O.observation(output):
        yield


def execute():
    check_source(); sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    for path in (INPUT, CALLER) + tuple('src/trottertracks/'+n+'.py' for n in ADAPTED):
        if (ROOT/path).read_bytes() != R.blob(sha, path):
            raise ValueError('Unfixed execution SOURCE: '+path)
    inputs = json.loads((ROOT/INPUT).read_text())
    if inputs != build_inputs(): raise ValueError('Fixed LD3 input changed')
    output = ROOT/DIRECTORY/'run01'; output.mkdir(); (output/LABEL).mkdir()
    source = dict(commit=sha, path=INPUT, caller=CALLER)
    R.write_new(output/'EXECUTION.json', dict(source=source, started_UTC=R.utc(), maximum_scientific_calls=1))
    with installed(output):
        result = M.supervise(inputs['cases'], inputs, source, output)
    exits = [c for p in output.glob('PROCESS_EXITS_*.jsonl') for line in p.read_text().splitlines()
             for c in json.loads(line)['children']]
    gate = bool(exits) and all(c['exitcode'] == 0 for c in exits)
    R.write_new(output/'SUPERVISION.json', dict(**result,
        process_exit_gate=dict(recorded_children=len(exits), all_exit_codes_zero=gate)))
    status = ('UNKNOWN' if result['errors'] or result['monitor_errors'] or not gate else
              result['receipts'][0]['status'])
    R.write_new(output/'STOP.json', dict(status=status, source=source, finished_UTC=R.utc(),
        mandatory_stop=True, automatic_retry=False, sufficient_N=None, G=None, G_qualified=None))
    print('STOP:', status, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'execute'))
    (prepare if parser.parse_args().mode == 'prepare' else execute)()
