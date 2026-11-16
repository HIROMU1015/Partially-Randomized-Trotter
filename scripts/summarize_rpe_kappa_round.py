"""Per-L_D kappa table of an incumbent-scope-CONFIGURATION round (arithmetic on saved outputs only).

Reads ROUND.json and ROUND_RESULT.json -- or, for a round stopped by a search-policy change, ROUND_RESULT_INTERRUPTED.json --
plus the route's configuration table and the published MEAN_HANDOFF packets (git), and C's catalog through the optimizer.
Writes KAPPA_COMPARISON.json and KAPPA_COMPARISON.csv.  No Q3, price, signal or compile.

    python summarize_rpe_kappa_round.py --problem P.json --run-dir DIR --output DIR
"""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_method_optimizer as O  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def git_json(loc):
    value = json.loads(subprocess.check_output(['git', 'show', loc['commit'] + ':' + loc['path']], cwd=ROOT))
    for key in loc.get('selector', []):
        value = value[key]
    return value


def upper(x):
    """Exact bound, or the outward upper end of a compact 30-digit interval."""
    if isinstance(x, dict) and 'upper_30' in x:
        m, e = x['upper_30'].split('e')
        return Fraction(int(m)) * Fraction(10) ** int(e)
    return None if x is None else O.frac(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', type=Path, required=True);ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args();run = a.run_dir;problem = O.load_json(a.problem);tau = O.tau(problem)
    spec = O.load_json(run / 'ROUND.json')
    path = run / 'ROUND_RESULT.json' if (run / 'ROUND_RESULT.json').exists() else run / 'ROUND_RESULT_INTERRUPTED.json'
    full = O.load_json(path);res = full['methods']['PR_PAULI']
    rule = spec['routes_override']['PR_PAULI'][0];table = rule['configurations']
    variant = {}
    for r in res['generation']['variants']:
        cid = r.get('physically_identical_to') or r.get('configuration_id')
        variant.setdefault(cid, dict(L_D=r['L_D'], kappa_multiplier=r['variant']['kappa_multiplier'],
                                     rounds_differing_from_base=r.get('rounds_differing_from_base') or []))
    _, _, catalog = O.load_catalog(ROOT, problem)
    saved_pass = [p for p in catalog['candidates'] if p['method'] == 'PR_PAULI' and p['q3_status'] == O.PASS and p['conditional_cost'] is not None]
    saved_min = min(saved_pass, key=lambda p: O.frac(p['conditional_cost']))
    not_adopted = (res.get('interrupted') or {}).get('computed_not_adopted', [])
    rows = []
    for b in res['by_configuration']:
        cid = b['configuration_id'];v = variant.get(cid, {});trace = [t for t in res['trace'] if t['configuration_id'] == cid]
        packet = git_json(table[cid]['signal_packet'])['packets'][0] if cid in table else None
        best = b.get('cheapest_feasible_evaluated') or (b['best'] and dict(N_X=O.split(b['best']['vector'])[0], N_Y=O.split(b['best']['vector'])[1],
                                                                             cost=b['best']['cost'], MSE_upper=b['best']['MSE_upper'], adopted_by_rule=True))
        ev = b.get('evaluated') or {}
        rows.append(dict(L_D=v.get('L_D', b['key'].get('L_D')), kappa_multiplier=v.get('kappa_multiplier', '1'), configuration_id=cid,
                         changed_rounds=v.get('rounds_differing_from_base', []),
                         signals_reused=None if packet is None else len(packet['reused_rounds']),
                         signals_new=None if packet is None else len(packet['new_rounds']),
                         saved_Q3_points=ev.get('saved'), rule_steps=len(trace),
                         rule_new_Q3=sum(1 for t in trace if t['answer_source'] != 'SAVED'),
                         lookahead_not_adopted_Q3=sum(1 for x in not_adopted if x['configuration_id'] == cid),
                         evaluated_allocations=ev.get('total'), feasible=ev.get('feasible'), infeasible=ev.get('infeasible'),
                         stop_reason=b['stop_reason'],
                         best_N_X=None if best is None else best['N_X'], best_N_Y=None if best is None else best['N_Y'],
                         best_cost=None if best is None else O.frac(best['cost']),
                         best_MSE_upper_over_tau=None if best is None else float(upper(best['MSE_upper']) / tau),
                         best_adopted_by_rule=None if best is None else best.get('adopted_by_rule')))
    for r in rows:
        ref = next((x for x in rows if x['L_D'] == r['L_D'] and Fraction(x['kappa_multiplier']) == 1), None)
        ok = ref and ref['best_cost'] is not None and r['best_cost'] is not None
        r['cost_minus_kappa1'] = r['best_cost'] - ref['best_cost'] if ok else None
        r['cost_over_kappa1'] = float(r['best_cost'] / ref['best_cost']) if ok else None
    rows.sort(key=lambda r: (r['L_D'], Fraction(r['kappa_multiplier'])))
    feasible = [r for r in rows if r['best_cost'] is not None]
    new_min = min(feasible, key=lambda r: r['best_cost']) if feasible else None
    out = dict(schema='RPE_KAPPA_ROUND_COMPARISON_V1', round_id=spec['round_id'], result_file=path.name, tau=tau, method_stop=res['stop_reason'],
               interpretation=('values at the stop of a round interrupted by a search-policy change: every configuration was still improving; '
                               'kappa differences are stop-time incumbents, not converged optima') if 'INTERRUPTED' in path.name else 'round result',
               saved_PR_Pauli_minimum=dict(label=saved_min['label'], configuration_id=saved_min['configuration_id'], cost=O.frac(saved_min['conditional_cost'])),
               round_minimum=None if new_min is None else dict(L_D=new_min['L_D'], kappa_multiplier=new_min['kappa_multiplier'],
                                                               configuration_id=new_min['configuration_id'], cost=new_min['best_cost'], N=new_min['best_N_X']),
               saved_minimum_updated=None if new_min is None else new_min['best_cost'] < O.frac(saved_min['conditional_cost']),
               rows=rows, not_compared=[dict(candidate=c['candidate'], needs=c['needs']) for c in res.get('not_ready_candidates', [])],
               science=dict(price='PriceRules MODEL (D2 c1f95e08), no fit', compile=0, this_script='arithmetic only'))
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'KAPPA_COMPARISON.json').write_text(json.dumps(O.jsonable(out), indent=1, ensure_ascii=False) + '\n')
    keys = ['L_D', 'kappa_multiplier', 'configuration_id', 'changed_rounds', 'signals_reused', 'signals_new', 'saved_Q3_points', 'rule_steps',
            'rule_new_Q3', 'lookahead_not_adopted_Q3', 'evaluated_allocations', 'feasible', 'infeasible', 'stop_reason', 'best_N_X', 'best_N_Y',
            'best_cost', 'best_MSE_upper_over_tau', 'best_adopted_by_rule', 'cost_minus_kappa1', 'cost_over_kappa1']
    with (a.output / 'KAPPA_COMPARISON.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n');w.writerow(keys)
        for r in rows:
            w.writerow([O.show(r[k]) if isinstance(r[k], Fraction) else (' '.join(map(str, r[k])) if isinstance(r[k], list) else r[k]) for k in keys])
    for r in rows:
        print(r['L_D'], r['kappa_multiplier'], r['configuration_id'][:8], r['stop_reason'], r['best_N_X'], r['best_cost'] and O.show(r['best_cost']),
              '%.9f' % r['best_MSE_upper_over_tau'], r['cost_over_kappa1'], 'steps', r['rule_steps'], 'new', r['rule_new_Q3'],
              'not_adopted', r['lookahead_not_adopted_Q3'], 'evaluated', r['evaluated_allocations'])
    print('saved min', O.show(out['saved_PR_Pauli_minimum']['cost']), 'round min', out['round_minimum'] and O.show(out['round_minimum']['cost']),
          'updated', out['saved_minimum_updated'])


if __name__ == '__main__':
    main()
