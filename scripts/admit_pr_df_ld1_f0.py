"""Connect saved F0 diagnostics to existing C axes; no probability/price arithmetic."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from trottertracks import price_implementation_guard as G
from admit_pr_df_ld1_q3_prices import at, equal, require

ROOT = Path(__file__).resolve().parents[1]
BASE = 'docs/research/pr_df_ld1_f0_admission_20261115'
D_DIR = 'docs/research/d_pr_df_ld1_f0_law_diagnostic_20261115/'
D = 'd749ef92e8ac3087747481a46aeb28ec4ca04c4b'
EXEC = 'fad0c1e4fe41c0e4c6d0690158056572295ed15a'
RAW = 'ec78fde57e117e9bbf79a26bfaa2e1575c1f5784'
HANDOFF = dict(commit=D, path=D_DIR+'C_HANDOFF.json')
INPUTS = dict(commit=EXEC, path=D_DIR+'INPUTS.json', selector=[])
RESULT = dict(commit=RAW, path=D_DIR+'run01/RESULT.json')
PRIOR = dict(commit='c99d511e2422b8f250490b11054f754a81e536c5',
             path='docs/research/pr_df_ld1_q3_price_admission_20261115/ADMISSION.json')
FACTS = dict(commit=PRIOR['commit'], path='docs/research/pr_df_ld1_q3_price_admission_20261115/SOURCE_FACTS.json')
PLAN = dict(commit='950577f416c17a20d309a8053a4bb016027b724d',
            path='docs/research/d_pr_df_ld1_ec_plan_20261115/NEXT_F0_PLAN.json')
M4 = dict(commit='9edfa5d3201f22c23dd77569d102801be80dacf3',
          path='docs/research/pr_df_ld1_m4_holdout_admission_20261115/ADMISSION.json')
TARGET = 'ideal_finite_RTE_law'
MAP = 'NUMPY_FLOAT64_CUMSUM_NORMALIZED_SEARCHSORTED_RIGHT_53BIT_UNIFORM'


def check_table(t, inp, diagnostic, original, result_table, preparation_hash):
    equal((t['id'], inp['id'], diagnostic['id'], result_table['id']), (t['id'],)*4, 'Table identity')
    equal(diagnostic['status'], 'DIAGNOSTIC_COMPLETE', 'Saved diagnostic status')
    equal(result_table['status'], diagnostic['status'], 'Raw result status')
    equal(result_table['path'], t['diagnostic']['path'], 'Raw result locator')
    equal(diagnostic['kind'], inp['kind'], 'Diagnostic kind')
    equal(diagnostic['source'], inp['source'], 'Diagnostic/input original SOURCE')
    equal(diagnostic['map'], MAP, 'Conditional grid map')
    require(diagnostic['PRNG_independence_proved'] is False, 'No PRNG independence upgrade')
    require(diagnostic['physical_law_distance'] is None, 'Actual law distance remains unknown')
    require(diagnostic['qualified_for_compiled_price'] is False, 'Not a price qualification')
    if inp['kind'] == 'component':
        equal(inp['signature']['preparation_hash'], preparation_hash, 'C/F0 preparation hash')
        equal(original['hash'], preparation_hash, 'Original preparation hash')
        equal(original['components'], inp['signature']['components'], 'Saved ordered component metadata')
    else:
        for k, v in inp['signature'].items():
            equal(original[k], v, 'Saved order metadata '+k)


def check_round(bound, order_input, assumptions):
    r = order_input['row']; s = order_input['signature']
    equal((bound['m'], bound['physical_id'], bound['event_positions']),
          (r['m'], r['physical_id'], r['saved_event_positions']), 'Bound round/physical/event positions')
    equal((s['m'], s['candidate_id'], s['q_PF'], s['R_outer']),
          (r['m'], r['physical_id'], r['q'], r['R_outer']), 'F0 schedule identity')
    equal(bound['axes'], r['axes'], 'Shared X/Y bound')
    equal(bound['source'], order_input['source'], 'Bound law SOURCE')
    equal(bound['order_table'], order_input['id']+'.json', 'Bound order table')
    equal(bound['component_table'], 'components.json', 'Shared component table')
    equal(bound['assumptions'], assumptions, 'Saved bound assumptions')
    for k in ('actual_PCG64_joint_distance', 'price_oscillation', 'price_shift', 'qualified_E_C'):
        require(bound[k] is None, 'Keep unknown '+k)


def check_axis(row, old, order_input, law):
    m = row['m']; p = old['physical_candidate']; r = order_input['row']
    equal((row['m'], row['axis'], row['physical_id']),
          (old['m'], old['axis'], old['physical_id']), 'C/F0 round/axis/physical identity')
    equal((r['m'], r['physical_id'], r['q'], r['R_outer']),
          (m, p['id'], p['q'], p['R_outer']), 'C/F0 physical schedule')
    equal((p['system'], p['method'], p['formula'], p['prefix'], p['delta'], p['K']),
          ('H4', 'PR', '2nd', 1, 0.1, 2), 'Fixed H4 PR-DF LD1 preparation')
    require(row['axis'] in r['axes'], 'F0 shared axis coverage')
    equal(row['expectation_target'], TARGET, 'Q3 ideal-law target unchanged')
    equal(law['definition']['model'], TARGET, 'Structural ideal-law target')
    decl = law['definition']['A_declaration']
    equal((decl['candidate_id'], decl['q_PF'], decl['R_outer'], decl['K']),
          (p['id'], p['q'], p['R_outer'], p['K']), 'Saved structural law identity')
    require(old['structural']['compiled_price'] is False, 'E[S] is not E[C]')
    for k in ('actual_PCG64_joint_distance', 'price_correction', 'qualified_E_C'):
        require(row[k] is None, 'F0 axis unknown remains null '+k)


def build(root, reader=None):
    read = reader or (lambda ref: G.evidence(root, ref))
    h, inputs, result, prior, facts = [read(r) for r in (HANDOFF, INPUTS, RESULT, PRIOR, FACTS)]
    equal(h['execution_source'], EXEC, 'F0 execution SOURCE')
    equal(h['raw_publication_commit'], RAW, 'F0 raw publication')
    equal(h['inputs'], INPUTS, 'F0 fixed inputs')
    equal(h['adopted_plan'], PLAN, 'Adopted F0 plan')
    equal(inputs['plan'], PLAN, 'Input adopted plan')
    equal((h['status'], result['status']), ('F0_DIAGNOSTIC_COMPLETE',)*2, 'Completed F0')
    equal(h['counts'], result['counts'], 'Raw/public completion counts')
    equal(inputs['target_cost_law'], TARGET, 'F0 target unchanged')
    equal(len(h['tables']), 14, 'One component and thirteen order tables')
    equal({t['id'] for t in h['tables']}, {'components'} | {'order_m'+str(m) for m in range(13)}, 'Unique F0 tables')
    equal({(r['m'], r['axis']) for r in h['rows']}, {(m,a) for m in range(13) for a in ('X','Y')}, 'Unique 26 F0 axes')
    equal(len(h['rows']), 26, 'No duplicated axis rows')
    equal(len(prior['rows']), 26, 'Existing C 26 axes')
    tables = []; by_id = {}; input_by_id = {}
    for j, t in enumerate(h['tables']):
        entry = dict(id=t['id'], source=at(HANDOFF, 'tables', j), diagnostic=t['diagnostic'], input=t['input'], status='HELD', reason=None)
        try:
            equal(t['diagnostic']['commit'], RAW, 'Raw table publication')
            equal((t['input']['commit'], t['input']['path']), (EXEC, INPUTS['path']), 'Input SOURCE')
            inp = read(t['input']); diagnostic = read(t['diagnostic']); original = read(inp['source'])
            raw_table = next(r for r in result['tables'] if r['id'] == t['id'])
            check_table(t, inp, diagnostic, original, raw_table, facts['preparation_hash'])
            entry.update(status='ACCEPTED_SAVED_DIAGNOSTIC', original_source=deepcopy(inp['source']),
                saved_summary={k:deepcopy(diagnostic[k]) for k in ('TV', 'triangle_bound', 'map', 'PRNG_independence_proved', 'physical_law_distance', 'qualified_for_compiled_price')},
                support_changes=at(t['diagnostic'], 'support_changes'))
            input_by_id[t['id']] = inp
        except (KeyError, ValueError, OSError, StopIteration) as error:
            entry['reason'] = str(error)
        by_id[t['id']] = len(tables); tables.append(entry)
    bounds = []; rows = []
    for m in range(13):
        dr = next(r for r in h['rows'] if r['m'] == m and r['axis'] == 'X')
        b = dict(m=m, source=deepcopy(dr['conditional_round_bound']), status='HELD', reason=None)
        try:
            for name in ('components', 'order_m'+str(m)):
                require(tables[by_id[name]]['status']=='ACCEPTED_SAVED_DIAGNOSTIC', 'Required diagnostic held '+name)
            bound = read(b['source']); inp = input_by_id['order_m'+str(m)]
            check_round(bound, inp, h['assumptions'])
            b.update(status='ACCEPTED_SAVED_CONDITIONAL_BOUND', saved_bound=deepcopy(bound), shared_axes=['X','Y'])
        except (KeyError, ValueError, OSError) as error:
            b['reason'] = str(error)
        bounds.append(b)
    for j, dr in enumerate(h['rows']):
        m = dr['m']; axis = dr['axis']
        old_index, old = next((i,r) for i,r in enumerate(prior['rows']) if r['m']==m and r['axis']==axis)
        entry = {k:deepcopy(old[k]) for k in ('m','axis','physical_id','physical_candidate','configuration_id','total_configuration_id','original_canonical_id','total_candidate_id','alignment','Q3','original_law','total_law','structural')}
        entry.update(source=at(HANDOFF,'rows',j), prior_admission=at(PRIOR,'rows',old_index),
            status='HELD', reason=None, diagnostics=[['tables',by_id['components']],['tables',by_id['order_m'+str(m)]]],
            conditional_bound=['round_bounds',m], qualified_E_C=None, price_provider_record=False)
        try:
            require(bounds[m]['status']=='ACCEPTED_SAVED_CONDITIONAL_BOUND', 'Round bound held')
            inp = input_by_id['order_m'+str(m)]
            for k, wanted in [('component_diagnostic',tables[by_id['components']]['diagnostic']),
                              ('order_diagnostic',tables[by_id['order_m'+str(m)]]['diagnostic']),
                              ('input',tables[by_id['order_m'+str(m)]]['input']),
                              ('conditional_round_bound',bounds[m]['source'])]:
                equal(dr[k], wanted, 'Axis shared locator '+k)
            check_axis(dr, old, inp, read(old['structural']['law_source']))
            entry['status']='CONNECTED_SAVED_CONDITIONAL_LAW_DIAGNOSTIC'
        except (KeyError, ValueError, OSError) as error:
            entry['reason']=str(error)
        rows.append(entry)
    counts = {}
    for name, values in [('diagnostic_tables',tables),('round_bound_pairs',bounds),('candidate_axes',rows)]:
        n = sum(r['status']!='HELD' for r in values)
        counts[name] = dict(requested=len(values), accepted=n, held=len(values)-n)
    return dict(schema='PR_DF_LD1_F0_C_CONSUMER_ADMISSION_V1', sources=dict(handoff=HANDOFF,inputs=INPUTS,raw_result=RESULT,adopted_plan=PLAN,prior_admission=PRIOR,m4_preserved=M4),
        status='SAVED_DIAGNOSTICS_CONNECTED' if all(c['held']==0 for c in counts.values()) else 'PARTIAL_ADMISSION',
        counts=counts, tables=tables, round_bounds=bounds, rows=rows,
        saved_stage_summary=deepcopy(h['stage_summary']), conditional_path_scope=deepcopy(h['conditional_path_scope']),
        assumptions=deepcopy(h['assumptions']), remaining_evidence=deepcopy(h['remaining_evidence']),
        distribution_scope='P is the saved reference-weight law; G is the uniform 53-bit grid map. This is not a bound on actual PCG64 joint streams.',
        actual_PCG64_joint_distance=None, full_support_residual_oscillation=None, price_correction=None, qualified_E_C=None, CI=None,
        price_provider_records_created=0, price_guard_changed=False, scientific_reanalysis=False, F0_rerun=False,
        saved_E_I_S_modified=False, frozen_price_model_modified=False, m4_independent_holdout_modified=False, Q3_ideal_contract_modified=False,
        N=None,G=None,winner=None,mandatory_stop=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--out',type=Path,required=True); args=parser.parse_args()
    require(not args.out.exists(), 'Reuse existing fixed admission; do not overwrite')
    data=build(ROOT); args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'ADMISSION.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(status=data['status'],counts=data['counts'],new_scientific_calculations=0)))


if __name__ == '__main__': main()
