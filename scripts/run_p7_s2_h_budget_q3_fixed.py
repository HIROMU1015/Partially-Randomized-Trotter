"""Two adopted width-h allocations under the saved coarse p7/S2 cost budget."""
import argparse
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_p7_h_saved_caller', ROOT/'scripts/run_p7_s2_d01_same_shots_q3_fixed.py')
C = importlib.util.module_from_spec(spec); sys.modules[spec.name] = C; spec.loader.exec_module(C)
F, R, T, M = C.F, C.R, C.T, C.M
BASE = 'bec032fc26093177f529754060df06d5375944b4'
PRICE = 'ecf45508a63ddf75a4806b47179f5f5102fa90e0'
PRICE_PATH = 'docs/research/d_pf_p7_s2_h_comparison_prices_20261115/run01/COMPARISON_PRICES.json'
DIRECTORY = 'docs/research/p7_s2_h_budget_q3_results_20261116'
INPUT = DIRECTORY+'/INPUTS.json'
CALLER = 'scripts/run_p7_s2_h_budget_q3_fixed.py'
BUDGET = 435481090
RULES = ('A_SHAPE', 'B_PRICE')


def check_source():
    C.check_source()
    path = 'scripts/run_p7_s2_d01_same_shots_q3_fixed.py'
    if (ROOT/path).read_bytes() != R.blob(BASE, path):
        raise ValueError('Saved fine-width caller SOURCE differs')


def joined_prices(table, base, binding, view):
    """Source-bound price intake only; no price prediction/compile/guard bypass."""
    if (table['configuration_id'] != C.CID or base['configuration_id'] != C.CID
            or table['quantity'] != 'RZ_COUNT_PER_MEASURED_FULL_CONTROLLED_HADAMARD_SHOT_P0_EXCLUDED'
            or table['counts'] != dict(total=26, direct=10, extrapolated=16,
                missing_comparison_prices=0, independently_validated_long_q=0)
            or table['source_accuracy_join'] != F.ref(BASE, C.DIRECTORY+'/C_RESULT_JOIN.json')):
        raise ValueError('Fine-width price domain/source differs')
    rows = sorted(enumerate(table['rows']), key=lambda x: (x[1]['m'], x[1]['axis']))
    if [(r['m'], r['axis']) for _, r in rows] != [(m, a) for m in range(13) for a in ('X', 'Y')]:
        raise ValueError('Missing/duplicate price axes')
    for index, row in rows:
        m = row['m']; physical = base['physical_rows'][m]['physical_candidate']
        original = binding['original_configuration']['rounds'][m]['candidate']
        if (row['configuration_id'] != C.CID or row['physical_id'] != physical['id']
                or row['physical_candidate'] != physical or row['original_canonical_id'] != original['id']
                or T.B.a.exact_scalar(row['T']) != T.B.a.exact_scalar(original['T']) or row['q'] != original['q']
                or T.B.a.exact_scalar(row['actual_delta']) != T.B.a.exact_scalar(original['delta'])
                or row['T_binary64_hex'] != float(T.B.a.exact_scalar(original['T'])).hex()
                or row['actual_delta_binary64_hex'] != float(T.B.a.exact_scalar(original['delta'])).hex()
                or view['signal_packet']['means'][m]['provider_binding']['physical_context']['physical_candidate'] != physical):
            raise ValueError('Physical schedule/price join differs at '+str((m, row['axis'])))
        F.integer(row['unit_price_RZ'])
        if m <= 4:
            if (row['origin'] != 'DIRECT_COMPILED' or row['direct_evidence_status'] != 'ACCEPTED_DIRECT'
                    or row['direct_compiled_RZ'] != row['unit_price_RZ']
                    or row['direct_receipt'] is None or row['direct_sidecar'] is None):
                raise ValueError('Direct price evidence differs')
        elif (row['origin'] != 'ADOPTED_RULE_EXTRAPOLATION' or row['direct_receipt'] is not None
                or row['direct_compiled_RZ'] is not None or row['rule_error_bound_RZ'] is not None):
            raise ValueError('Extrapolation qualification differs')
    return rows


def build_inputs():
    check_source()
    prior = F.document(BASE, C.INPUT); base = prior['cases'][0]
    binding, view = F.metadata(base)
    table = F.document(PRICE, PRICE_PATH); rows = joined_prices(table, base, binding, view)
    cases, proposals, seen = [], [], {}
    saved_vector = tuple(n for r in base['allocation']['shots'] for n in (r['N_X'], r['N_Y']))
    options = deepcopy(prior['options'])
    for rule in RULES:
        generated = F.generate([F.N0[r['m']] for _, r in rows],
            [r['unit_price_RZ'] for _, r in rows], BUDGET, rule)
        if generated is None:
            proposals.append(dict(rule=rule, disposition='BUDGET_MENU_EMPTY', scientific_calls=0)); continue
        case = deepcopy(base)
        case.update(label='B0I_P7_S2_H__'+rule, rule=rule,
            allocation=dict(allocation_id='B0I_P7_S2_H_'+rule+'_COARSE_BUDGET_01', shot_policy='USER_SUPPLIED',
                shots=[dict(m=m, N_X=generated['shots'][2*m], N_Y=generated['shots'][2*m+1]) for m in range(13)]),
            generation={k:v for k,v in generated.items() if k != 'shots'},
            conditional_cost=generated['conditional_cost'], unused_budget=generated['unused_budget'],
            price_terms=[dict(m=r['m'], axis=r['axis'], physical_id=r['physical_id'],
                original_candidate_id=r['original_canonical_id'], N=n, C=r['unit_price_RZ'],
                N_times_C=n*r['unit_price_RZ'], origin=r['origin'],
                price_source=F.ref(PRICE, PRICE_PATH, ['rows', index]),
                direct_receipt=r['direct_receipt'], direct_sidecar=r['direct_sidecar'],
                rule_id=r['rule_id'], rule_error_bound_RZ=r['rule_error_bound_RZ'])
                for (index,r),n in zip(rows, generated['shots'])])
        direct = sum(t['N_times_C'] for t in case['price_terms'] if t['origin']=='DIRECT_COMPILED')
        extra = sum(t['N_times_C'] for t in case['price_terms'] if t['origin']=='ADOPTED_RULE_EXTRAPOLATION')
        assert direct+extra == case['conditional_cost'] <= BUDGET
        case['price_partition'] = dict(direct_RZ=direct, extrapolated_RZ=extra,
            extrapolated_fraction=F.exact(Fraction(extra, direct+extra)))
        bounds = F.count_bounds(case['allocation']['shots']); case['finite_scope_preflight'] = bounds
        if (bounds['maximum_round_outcomes'] > options['maximum_round_outcomes']
                or bounds['verification_pairs_upper'] > options['maximum_verification_pairs']
                or bounds['transition_pairs_upper'] > options['total_state_refinement']['maximum_transition_pairs']):
            raise ValueError('Concrete count-only scope cannot be contained')
        vector = tuple(generated['shots'])
        disposition, reused = ('REUSE_SAVED_FINE_Q3', F.ref(BASE, C.DIRECTORY+'/run01/'+C.LABEL+'/API_RETURN.json')) if vector==saved_vector else (
            ('SAME_VECTOR_AS_OTHER_RULE', seen[vector]) if vector in seen else ('NEW_EVALUATION', None))
        proposals.append(dict(case, disposition=disposition, reused_result=reused))
        if disposition=='NEW_EVALUATION':
            cases.append(case); seen[vector] = case['label']
    assert len(cases) <= 2
    contract = deepcopy(prior['statistical_contract']); contract['id'] = 'P7_S2_H_TWO_FIXED_COARSE_BUDGET_ALLOCATIONS_Q3'
    options['model_loss_contract']['scope'] = 'P7_S2_H_TWO_ADOPTED_FIXED_ALLOCATIONS_ONLY'
    concurrency = min(len(cases), max(1,len(os.sched_getaffinity(0))//(prior['acceleration']['workers_per_configuration']+1)))
    return dict(cases=cases, proposals=proposals, maximum_scientific_calls=2, actual_new_call_limit=len(cases),
        concurrent_configurations=concurrency, caps_N0=F.N0, budget_RZ=BUDGET,
        fixed_prices=F.ref(PRICE, PRICE_PATH), fixed_fine_Q3=F.ref(BASE,C.DIRECTORY+'/SUMMARY.json'),
        coarse_comparison=deepcopy(prior['comparison']),
        kernel_source=prior['kernel_source'], C_source=prior['C_source'],
        acceleration=deepcopy(prior['acceleration']), statistical_contract=contract, options=options,
        adopted_evaluation=deepcopy(prior['adopted_evaluation']),
        price_policy='TEN_DIRECT_SIXTEEN_ADOPTED_EXTRAPOLATION_MODEL_UNCERTAINTY_UNKNOWN',
        duplicate_check='SEE_SCOPE_AND_PREFLIGHT.md', signal_recomputed=False,
        sufficient_N=None, global_RMSE_bound=None, G=None, G_qualified=None, winner=None)


def prepare():
    inputs = build_inputs(); (ROOT/DIRECTORY).mkdir(exist_ok=True); R.write_new(ROOT/INPUT, inputs)
    for case in inputs['proposals']:
        print(case['rule'], case['disposition'], case.get('conditional_cost'), case.get('allocation'), flush=True)


def execute():
    check_source(); sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT, CALLER):
        if (ROOT/path).read_bytes()!=R.blob(sha,path): raise ValueError('Caller/INPUTS not fixed')
    inputs = json.loads((ROOT/INPUT).read_text())
    if inputs != build_inputs() or len(inputs['cases']) > 2: raise ValueError('Fixed two-rule inputs differ')
    output = ROOT/DIRECTORY/'run01'; output.mkdir()
    for case in inputs['cases']: (output/case['label']).mkdir()
    source = dict(commit=sha,path=INPUT,caller=CALLER)
    R.write_new(output/'EXECUTION.json',dict(source=source,started_UTC=R.utc(),maximum_scientific_calls=2,
        actual_new_call_limit=len(inputs['cases'])))
    with F.installed_workers(output): result = M.supervise(inputs['cases'],inputs,source,output)
    exits = [c for p in output.glob('PROCESS_EXITS_*.jsonl') for line in p.read_text().splitlines()
        for c in json.loads(line)['children']]
    exit_gate = dict(recorded_children=len(exits), all_exit_codes_zero=all(c['exitcode']==0 for c in exits))
    status = ('FIXED_SCIENTIFIC_WORKERS_SAVED' if not result['errors'] and not result['monitor_errors']
        and exit_gate['all_exit_codes_zero'] and len(result['receipts'])==len(inputs['cases'])
        and all(r['status']!='UNKNOWN' for r in result['receipts']) else 'UNKNOWN_STOP_NO_RETRY')
    R.write_new(output/'COMPLETION.json',dict(status=status,**result,process_exit_gate=exit_gate,
        finished_UTC=R.utc(),mandatory_stop=True,sufficient_N=None,G=None,G_qualified=None))
    print('Finished:',status,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=('prepare','execute'))
    (prepare if parser.parse_args().mode=='prepare' else execute)()
