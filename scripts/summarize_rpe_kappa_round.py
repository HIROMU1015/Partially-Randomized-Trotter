"""Per-L_D kappa comparison table of an approved incumbent-scope-CONFIGURATION round (arithmetic on saved outputs only).

Reads the round directory (ROUND.json, ROUND_RESULT.json, jobs/), the route's configuration table and the
published MEAN_HANDOFF packets (git), and C's catalog through the optimizer.  Writes KAPPA_COMPARISON.json and
KAPPA_COMPARISON.csv.  No Q3, price, signal or compile.

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', type=Path, required=True);ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args();run = a.run_dir;problem = O.load_json(a.problem);tau = O.tau(problem)
    spec = O.load_json(run / 'ROUND.json');res = O.load_json(run / 'ROUND_RESULT.json')['methods']['PR_PAULI']
    rule = spec['routes_override']['PR_PAULI'][0];table = rule['configurations']
    gen = res['generation'];variant = {}
    for r in gen['variants']:
        cid = r.get('physically_identical_to') or r.get('configuration_id')
        variant.setdefault(cid, dict(L_D=r['L_D'], kappa_multiplier=r['variant']['kappa_multiplier'], formula=r['variant']['formula'],
                                     delta_over_h=r['variant']['delta_over_h'], disposition=r['disposition'],
                                     rounds_differing_from_base=r.get('rounds_differing_from_base') or [],
                                     base_configuration_id=r['base_configuration_id']))
    _, _, catalog = O.load_catalog(ROOT, problem)
    saved_pass = [p for p in catalog['candidates'] if p['method'] == 'PR_PAULI' and p['q3_status'] == O.PASS and p['conditional_cost'] is not None]
    saved_min = min(saved_pass, key=lambda p: O.frac(p['conditional_cost']))
    jobs = {}
    for req in (run / 'jobs' / 'PR_PAULI').glob('*/REQUEST.json'):
        q = json.loads(req.read_text());out = req.parent / 'OUTCOME.json'
        jobs[(q['configuration_id'], tuple(O.interleave(q['N_X'], q['N_Y'])))] = dict(
            job=req.parent.name, status=json.loads(out.read_text())['status'] if out.exists() else 'NO_OUTCOME')
    rows = []
    for b in res['by_configuration']:
        cid = b['configuration_id'];v = variant.get(cid, {});trace = [t for t in res['trace'] if t['configuration_id'] == cid]
        packet = git_json(table[cid]['signal_packet'])['packets'][0] if cid in table else None
        rule_new = [t for t in trace if t['answer_source'] == 'NEW_Q3_JOB']
        rule_vectors = {tuple(t['vector']) for t in trace}
        extra = [k for k in jobs if k[0] == cid and k[1] not in rule_vectors]
        saved_points = [e for e in res['evaluated'] if e['configuration_id'] == cid and e['source'] == 'SAVED']
        best = b['best']
        rows.append(dict(L_D=v.get('L_D', b['key'].get('L_D')), kappa_multiplier=v.get('kappa_multiplier', '1'), configuration_id=cid,
                         formula=v.get('formula'), delta_over_h=v.get('delta_over_h'),
                         changed_rounds=v.get('rounds_differing_from_base', []),
                         signals_reused=None if packet is None else len(packet['reused_rounds']),
                         signals_new=None if packet is None else len(packet['new_rounds']),
                         saved_Q3_points=len(saved_points), rule_steps=len(trace), rule_new_Q3=len(rule_new),
                         rule_reused_saved=sum(1 for t in trace if t['answer_source'] == 'SAVED'),
                         lookahead_extra_Q3=len(extra),
                         lookahead_extra_statuses=sorted({jobs[k]['status'] for k in extra}),
                         evaluated_allocations=len(saved_points) + len([t for t in trace if t['answer_source'] != 'SAVED']) + len(extra),
                         stop_reason=b['stop_reason'],
                         best_N_X=None if best is None else O.split(best['vector'])[0],
                         best_N_Y=None if best is None else O.split(best['vector'])[1],
                         best_cost=None if best is None else O.frac(best['cost']),
                         best_MSE_upper=None if best is None else O.frac(best['MSE_upper']),
                         best_MSE_upper_over_tau=None if best is None else float(O.frac(best['MSE_upper']) / tau),
                         best_source=None if best is None else best['source'], best_job=None if best is None else best.get('job')))
    for r in rows:
        ref = next((x for x in rows if x['L_D'] == r['L_D'] and Fraction(x['kappa_multiplier']) == 1), None)
        if ref and ref['best_cost'] is not None and r['best_cost'] is not None:
            r['cost_minus_kappa1'] = r['best_cost'] - ref['best_cost'];r['cost_over_kappa1'] = float(r['best_cost'] / ref['best_cost'])
        else:
            r['cost_minus_kappa1'] = r['cost_over_kappa1'] = None
    rows.sort(key=lambda r: (r['L_D'], Fraction(r['kappa_multiplier'])))
    feasible = [r for r in rows if r['best_cost'] is not None]
    new_min = min(feasible, key=lambda r: r['best_cost']) if feasible else None
    not_compared = [dict(candidate=c['candidate'], configuration_id=c['configuration_id'], needs=c['needs'], price_needs=c.get('price_needs'))
                    for c in res['not_ready_candidates']]
    not_compared += [dict(configuration_id=r['configuration_id'], L_D=r['L_D'], kappa_multiplier=r['kappa_multiplier'],
                          missing='NO_FEASIBLE_ALLOCATION_FOUND_BY_THE_RULE_IN_RANGE', stop_reason=r['stop_reason'])
                     for r in rows if r['best_cost'] is None]
    out = dict(schema='RPE_KAPPA_ROUND_COMPARISON_V1', round_id=spec['round_id'], tau=tau, method_stop=res['stop_reason'],
               saved_PR_Pauli_minimum=dict(label=saved_min['label'], configuration_id=saved_min['configuration_id'],
                                           cost=O.frac(saved_min['conditional_cost'])),
               round_minimum=None if new_min is None else dict(L_D=new_min['L_D'], kappa_multiplier=new_min['kappa_multiplier'],
                                                               configuration_id=new_min['configuration_id'], cost=new_min['best_cost'],
                                                               N=new_min['best_N_X']),
               saved_minimum_updated=None if new_min is None else new_min['best_cost'] < O.frac(saved_min['conditional_cost']),
               rows=rows, not_compared=not_compared, new_Q3_total=sum(r['rule_new_Q3'] + r['lookahead_extra_Q3'] for r in rows),
               science=dict(new_Q3='see rule_new_Q3 + lookahead_extra_Q3', price='PriceRules MODEL (D2 c1f95e08), no fit', compile=0))
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'KAPPA_COMPARISON.json').write_text(json.dumps(O.jsonable(out), indent=1, ensure_ascii=False) + '\n')
    keys = ['L_D', 'kappa_multiplier', 'configuration_id', 'changed_rounds', 'signals_reused', 'signals_new', 'saved_Q3_points',
            'rule_steps', 'rule_new_Q3', 'lookahead_extra_Q3', 'evaluated_allocations', 'stop_reason', 'best_N_X', 'best_N_Y',
            'best_cost', 'best_MSE_upper_over_tau', 'cost_minus_kappa1', 'cost_over_kappa1']
    with (a.output / 'KAPPA_COMPARISON.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n');w.writerow(keys)
        for r in rows:
            w.writerow([O.show(r[k]) if isinstance(r[k], Fraction) else (' '.join(map(str, r[k])) if isinstance(r[k], list) else r[k])
                        for k in keys])
    for r in rows:
        print(r['L_D'], r['kappa_multiplier'], r['configuration_id'][:8], r['stop_reason'], r['best_N_X'],
              r['best_cost'] and O.show(r['best_cost']), r['best_MSE_upper_over_tau'], r['cost_over_kappa1'],
              'steps', r['rule_steps'], 'new', r['rule_new_Q3'], 'extra', r['lookahead_extra_Q3'])
    print('saved min', O.show(out['saved_PR_Pauli_minimum']['cost']), 'round min', out['round_minimum'] and O.show(out['round_minimum']['cost']),
          'updated', out['saved_minimum_updated'])


if __name__ == '__main__':
    main()
